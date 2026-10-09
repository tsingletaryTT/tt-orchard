# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Check machines: `tt-orchard setup --check` for this box and `tt-orchard lab setup --check` for the lab,
read-only, as data for the web UI.

Both planners return steps (name, status, detail, actions). Here the actions are only described, as the
fix to run, and never applied: the page reports what is missing and the operator decides. The checks take
seconds to minutes (an import test, `tt-model list`, ssh), so the UI runs them as one background job at a
time (`ChecksJob`).
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from orchard import lab_setup, setup_machine

FIXABLE = (setup_machine.TODO, setup_machine.FAIL)


def _step_json(step) -> dict:
    return {"name": step.name, "status": step.status, "detail": step.detail,
            "fix": [a.describe() for a in step.actions] if step.status in FIXABLE else []}


def _coder_key(cfg) -> str:
    target = getattr(getattr(cfg, "coder", None), "target", None)
    return next((k for k, v in setup_machine.CODERS.items() if v["package"] == target), "coder-next")


def run_checks(cfg, *, env=None, setup_plan=None, lab_plan=None, run=None, config_path=None) -> list[dict]:
    """This box's readiness, then the lab's when the config has a [lab] table (whatever the mode). With
    `config_path` (the file the UI loaded), setup's config row reports that file instead of the checkout's,
    which setup itself would write."""
    env = env or setup_machine.default_env()
    ref = getattr(cfg, "reference_python", None)
    opts = setup_machine.Options(coder=_coder_key(cfg), runs_root=getattr(cfg, "runs_root", None),
                                 venv_dir=Path(ref).parent.parent if ref else None)
    with _PLAN_LOCK:                          # setup_machine keeps the gozer text it read in a module dict
        steps = (setup_plan or setup_machine.plan)(env, opts)
    if config_path is not None:
        steps = [setup_machine.Step("config", setup_machine.OK,
                                    f"this page uses {config_path} (mode {getattr(cfg, 'mode', 'local')})")
                 if s.name == "config" else s for s in steps]
    boxes = [{"name": "this box", "steps": [_step_json(s) for s in steps]}]
    lab = getattr(cfg, "lab", None)
    if lab is not None:
        run = run or setup_machine._run
        lab_ref = ref or (Path(lab.root) / "venvs" / "reference" / "bin" / "python")
        sources = [Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")]
        kw = dict(run=run, reference_python=lab_ref, local_fw=lab_setup.local_firmware(),
                  local_sfpi=lab_setup.local_sfpi(run), local_kmd=lab_setup.local_kmd(), models=(),
                  model_sources=sources, tt_model_root=getattr(cfg, "tt_model_root", None),
                  hf_home=getattr(cfg, "hf_home", None))
        lab_steps = (lab_plan or lab_setup.plan)(lab, **kw)
        boxes.append({"name": f"lab {lab.host}", "steps": [_step_json(s) for s in lab_steps]})
    return boxes


_PLAN_LOCK = threading.Lock()


class ChecksJob:
    """One check at a time, in a thread; `state()` is what the page polls."""

    def __init__(self, runner=run_checks, clock=time.time):
        self.runner, self.clock = runner, clock
        self._lock = threading.Lock()
        self._state: dict = {"running": False, "started": None, "finished": None, "boxes": [], "error": None}

    def start(self, cfg, **kw) -> bool:
        with self._lock:
            if self._state["running"]:
                return False
            self._state = {"running": True, "started": self.clock(), "finished": None, "boxes": [], "error": None}
        threading.Thread(target=self._work, args=(cfg, kw), daemon=True).start()
        return True

    def _work(self, cfg, kw) -> None:
        boxes, error = [], None
        try:
            boxes = self.runner(cfg, **kw)
        except Exception as exc:              # a check that crashes is reported, not raised into the server
            error = f"{type(exc).__name__}: {exc}"
        with self._lock:
            self._state.update(running=False, finished=self.clock(), boxes=boxes, error=error)

    def state(self) -> dict:
        with self._lock:
            return dict(self._state)
