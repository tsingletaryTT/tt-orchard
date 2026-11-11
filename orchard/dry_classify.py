# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A dry classification (issue 705): stage 0's measurement of a model against its nearest model, with nothing else started.

Stage 0 was a single point of failure before any chip work (issues 437, 537, 594, 647, 643): a misclassification showed only after the
coder had booted. `bringup --dry-run` and `run --dry-run` call this module so the operator sees the class, the route, the nearest model
and the reasons first.

What it does: it copies the shipped stage 0 template (orchard/skills/delta-triage-templates/delta_triage.py) into a scratch directory,
writes the template's `triage_config.json` there, runs the template with the current interpreter in isolated mode (`-I`) and a minimal
environment (no tokens, the Hugging Face hub offline), reads the `delta.json` it wrote and returns a short summary. The scratch
directory is removed on the way out. The template reads the two snapshot directories (headers of safetensors files, configs,
tokenizers); it writes only into its own directory.

What it does not do: start a coder, take a lease, open a device, run a container, write into a run directory, or use the network. It
imports none of the modules that do. The class and the path are the template's measurement. The stage 0 agent may still edit findings
in a real run, and stages that need a model (the stock-runtime path runs stage 0 as code with no template) do not go through here.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from orchard import safeio
from orchard.classes import CLASSES
from orchard.stages import SHIPPED_TEMPLATES

TEMPLATE_MAX_BYTES = 8 << 20
DELTA_MAX_BYTES = 16 << 20
DEFAULT_TIMEOUT_S = 600
MAX_ITEMS = 12                  # reasons and differences kept in the summary
MAX_TEXT = 400                  # characters kept of each reason, finding or problem
ID_MAX = 200
SIDE_FILES = ("audio_config_categories.json",)      # files the template reads from its own directory when present


def _line(value, what: str, limit: int = ID_MAX):
    """(text, None) for a short one-line string, else (None, problem)."""
    if not isinstance(value, str) or not value.strip() or len(value) > limit or any(c in value for c in "\0\n\r"):
        return None, f"the {what} must be one short line"
    return value, None


def _directory(value, what: str):
    if not isinstance(value, str) or not value or "\0" in value:
        return None, f"the {what} is not a directory"
    try:
        real = os.path.realpath(value)
        ok = os.path.isdir(real)
    except (ValueError, OSError):
        ok = False
    return (real, None) if ok else (None, f"the {what} is not a directory")


def _cut(text) -> str:
    return text if isinstance(text, str) and len(text) <= MAX_TEXT else (str(text)[:MAX_TEXT - 1] + "…")


def _problem(text: str) -> dict:
    return {"ok": False, "problem": _cut(text)}


def classify(*, model_id, model_snapshot, nearest_id, nearest_snapshot, bundle_dir=None, python: str | None = None,
             timeout_s: int = DEFAULT_TIMEOUT_S, runner=None) -> dict:
    """{"ok": True, "model", "nearest_model", "class", "path", "reasons", "differences"} or {"ok": False, "problem"}. Never raises for an
    input or template problem; the reason is in `problem` and holds no file content beyond the template's last error line."""
    model_id, why = _line(model_id, "model id")
    if why:
        return _problem(why)
    nearest_id, why = _line(nearest_id, "nearest model id")
    if why:
        return _problem(why)
    model_dir, why = _directory(model_snapshot, "model snapshot")
    if why:
        return _problem(why)
    nearest_dir, why = _directory(nearest_snapshot, "nearest snapshot")
    if why:
        return _problem(why)
    bundle = None
    if bundle_dir is not None:
        bundle, why = _directory(bundle_dir, "nearest bundle directory")
        if why:
            return _problem(why)
    try:
        template = safeio.read_bytes(SHIPPED_TEMPLATES["delta_triage.py"], max_bytes=TEMPLATE_MAX_BYTES)
    except (safeio.SafeReadError, OSError, ValueError):
        return _problem("the shipped stage 0 template cannot be read")
    with tempfile.TemporaryDirectory(prefix="orchard-dry-classify-") as scratch:
        run_dir = Path(scratch) / "run"
        stage = run_dir / "stages" / "0"            # the template insists on running from inside its run directory
        stage.mkdir(parents=True)
        try:
            safeio.atomic_write(stage / "delta_triage.py", template, mode=0o600)
            for name in SIDE_FILES:
                side = Path(SHIPPED_TEMPLATES["delta_triage.py"]).parent / name
                if side.is_file():
                    safeio.atomic_write(stage / name, safeio.read_bytes(side, max_bytes=TEMPLATE_MAX_BYTES), mode=0o600)
            import json
            config = {"run_dir": str(run_dir), "model_id": model_id, "model_snapshot": model_dir, "nearest_model_id": nearest_id,
                      "nearest_snapshot": nearest_dir, **({"nearest_bundle_dir": bundle} if bundle else {})}
            safeio.atomic_write(stage / "triage_config.json", json.dumps(config), mode=0o600)
        except (safeio.SafeReadError, OSError, ValueError):
            return _problem("the scratch directory cannot be prepared")
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": scratch, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
               "PYTHONDONTWRITEBYTECODE": "1", "LC_ALL": "C.UTF-8"}
        try:
            done = (runner or subprocess.run)([python or sys.executable, "-I", str(stage / "delta_triage.py")], cwd=str(stage), env=env,
                          capture_output=True, text=True, timeout=timeout_s, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return _problem(f"the stage 0 template did not finish within {timeout_s} s")
        except OSError as exc:
            return _problem(f"the stage 0 template could not start ({exc.__class__.__name__})")
        if getattr(done, "returncode", 1) != 0:
            last = [ln for ln in str(getattr(done, "stderr", "") or "").splitlines() if ln.strip()]
            return _problem("the stage 0 template failed: " + (last[-1] if last else f"exit {done.returncode}"))
        try:
            delta = safeio.read_json(stage / "delta.json", max_bytes=DELTA_MAX_BYTES, expect=dict)
        except (safeio.SafeReadError, OSError, ValueError):
            return _problem("the stage 0 template wrote no readable delta.json (it must hold a JSON object)")
    return _summary(delta, model_id, nearest_id)


def _summary(delta: dict, model_id: str, nearest_id: str) -> dict:
    cls = delta.get("class")
    cls = cls if isinstance(cls, str) and cls in CLASSES else "unknown"
    path = delta.get("path")
    reasons = [r for r in delta.get("path_reasons", []) if isinstance(r, str)] if isinstance(delta.get("path_reasons"), list) else []
    rows = []
    for d in delta.get("differences", []) if isinstance(delta.get("differences"), list) else []:
        if isinstance(d, dict) and isinstance(d.get("area"), str) and isinstance(d.get("finding"), str):
            rows.append({"area": _cut(d["area"]), "finding": _cut(d["finding"])})
    shown = lambda v, d: _cut(v) if isinstance(v, str) and v else d                      # noqa: E731
    return {"ok": True, "model": shown(delta.get("model"), model_id), "nearest_model": shown(delta.get("nearest_model"), nearest_id),
            "class": cls, "path": shown(path, "unknown"), "reasons": [_cut(r) for r in reasons[:MAX_ITEMS]], "differences": rows[:MAX_ITEMS]}


WIDTH = 80                      # the preflight page keeps every line inside 80 columns (tests/test_cli.py)


def _wrapped(label: str, text: str) -> list[str]:
    """`║  <label> <text>` wrapped to WIDTH columns, continuation lines indented under the text. Left bar only."""
    import textwrap
    head = f"║  {label} "
    return textwrap.wrap(text, width=WIDTH, initial_indent=head, subsequent_indent="║" + " " * (len(head) - 1),
                         break_long_words=True, break_on_hyphens=False) or [head.rstrip()]


def render(result: dict, *, command: str, would_block=()) -> str:
    """The text of a dry classification, left bars only (a right border breaks in a narrow terminal) and wrapped to 80 columns. The box is closed
    and the command follows it on one unframed line, so that it can be copied. `would_block` lists the machine-state blocks of the preflight
    ({"check", "reason", "detail"}) that would stop a real run and did not stop this one."""
    lines = []
    if not result.get("ok"):
        lines += _wrapped("stage 0 ", f"not classified: {result.get('problem', 'unknown problem')}")
    else:
        lines += _wrapped("stage 0 ", f"class {result['class']}   route {result['path']}")
        lines += _wrapped("model   ", result["model"])
        lines += _wrapped("nearest ", result["nearest_model"])
        for r in result["reasons"]:
            lines += _wrapped("reason  ", r)
        for d in result["differences"]:
            lines += _wrapped(f"{d['area'][:8]:<8}", d["finding"])
    for w in would_block:
        lines += _wrapped("would block a real run", f"{w['reason']}: {w['detail']}")
    # The command is printed below the box, unframed and unwrapped, so that it can be copied as it is.
    return "\n".join(lines + ["╚══", f"command   {command}"])
