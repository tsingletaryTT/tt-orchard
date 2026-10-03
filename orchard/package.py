"""Stage 7: stage a v6 thin package of a weights-only model, as supervisor code (plan 5).

This module owns the package a run hands to the operator. It never publishes. It runs one external
program on its own, `tt-model package-thin`, always with `--out` and never with a repo id
(`assert_no_publish`), and it runs the staged package's `install.sh` in a separate copy. It never
calls `tt-model push` or `publish`, `hf upload`, `git push` or the Hugging Face hub API. The
publish commands it writes are text for the operator (`publish_commands`).

The package is built from the nearest model's installed v6 bundle (the "source bundle") with the
same model code, wheels, environment and fixed vLLM arguments, and `--weights` naming the new
model at its pinned revision. Three edits follow, each found by failing first on this machine:

1. Fixed vLLM arguments. package-thin has no flag for them, so they are copied from the source
   bundle's run.sh and spliced in before the passed-through `"$@"` (`extra_args_from`,
   `splice_extra_args`).
2. Weights. The TT runtime takes its weights directory from MODEL_WEIGHTS_DIR, then HF_MODEL, then
   the config path. The bundle's model class is registered for the nearest model's architecture
   (a vision-language config), and the new model's config.json names a text-only architecture,
   which vLLM refuses. So run.sh runs `prepare_model_dir.py` before vLLM: it builds `model-dir/`
   from the nearest model's config files (shipped in `base_config/`) and the new model's tokenizer
   and weights. `--model`, HF_MODEL and MODEL_WEIGHTS_DIR all name that directory
   (`wire_weights`). `weights_wiring_problems` checks the result, and the gate runs it again.
3. Provenance. The manifest's `producer.hostname` is replaced, run.sh and install.sh are made
   executable (package-thin leaves them without +x), and every wheel must be byte-identical to
   the source bundle's, because wheels are binary and the scrub does not read them.

Stage 7 ships no tensor cache and no weights (orchard/scrub.py, scrub_package). The boot check
(`prepare_verify`, package_templates/verify_bundle.py) installs and serves a copy, so the staged
directory never gains a venv, a model-dir or a cache. The supervisor passes the agent shells'
environment to every command here (no tokens, HOME inside the run directory), so a call that
tried to upload would also find no credentials.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from orchard.defaults import (PACKAGE_HEALTH_TIMEOUT_S, PACKAGE_INSTALL_TIMEOUT_S,
                              PACKAGE_THIN_TIMEOUT_S, PACKAGE_VERIFY_DEADLINE_S)
from orchard.package_card import CardFacts, Number, non_commercial, read_license, render_card
from orchard.scrub import scrub_package
from orchard.stages import gate_weights_swap

TEMPLATES = Path(__file__).with_name("package_templates")
SWAP_TEMPLATES = Path(__file__).with_name("skills") / "weights-swap-templates"
BASE_CONFIG_FILES = ("config.json", "preprocessor_config.json", "video_preprocessor_config.json")

MODEL_DIR = '"$HERE/model-dir"'
WIRING_LINES = (f"export HF_MODEL={MODEL_DIR}", f"export MODEL_WEIGHTS_DIR={MODEL_DIR}")
PREPARE_LINE = '"$PYBIN" "$HERE/prepare_model_dir.py"'
EXEC_LINE = 'exec "${CMD[@]}"'
PASS_THROUGH = ' "$@")'


class PackageError(Exception):
    """Stage 7 cannot stage or check the package. The message says why."""


# ---- run.sh edits -------------------------------------------------------------------------------

def _cmd_line(text: str) -> str:
    lines = [ln for ln in text.splitlines() if ln.startswith("CMD=(")]
    if len(lines) != 1:
        raise PackageError(f"expected exactly one CMD=( line in run.sh, found {len(lines)}")
    return lines[0]


def extra_args_from(source_run_sh: str) -> str:
    """The fixed vLLM arguments a source bundle's run.sh adds after the generated ones.

    package-thin writes `--max_model_len N` last and the passed-through `"$@"` at the end, so
    whatever stands between the two was spliced in by the bundle's author."""
    m = re.search(r'--max_model_len \d+(?P<extra>.*) "\$@"\)\s*$', _cmd_line(source_run_sh))
    if m is None:
        raise PackageError('the source run.sh command has no "--max_model_len N ... "$@")" to read '
                           "the fixed vLLM arguments from")
    return m.group("extra").strip()


def splice_extra_args(text: str, extra: str) -> str:
    """Put `extra` just before the passed-through arguments, so an operator's own arguments still
    come last and win."""
    if not extra:
        return text
    line = _cmd_line(text)
    if not line.endswith(PASS_THROUGH):
        raise PackageError('the run.sh command does not end with "$@")')
    return text.replace(line, line[:-len(PASS_THROUGH)] + " " + extra + PASS_THROUGH)


def _once(pattern: str, repl: str, text: str, what: str, flags=0) -> str:
    out, n = re.subn(pattern, lambda m: repl, text, flags=flags)
    if n != 1:
        raise PackageError(f"expected exactly one {what} in run.sh, found {n}")
    return out


def wire_weights(text: str, model_id: str) -> str:
    """Point every weights setting in a generated run.sh at the bundle's model-dir."""
    q = re.escape(model_id)
    text = _once(rf'--model (?:"{q}"|{q})(?=\s)', f"--model {MODEL_DIR}", text, f'--model "{model_id}"')
    text = _once(r" --revision [0-9a-f]{40}(?=\s)", "", text, "--revision <40 hex>")
    text = _once(r" --tokenizer-revision [0-9a-f]{40}(?=\s)", "", text, "--tokenizer-revision <40 hex>")
    text = _once(r"^export HF_MODEL=.*$", "\n".join(WIRING_LINES), text, "export HF_MODEL= line",
                 flags=re.M)
    return _once(r"^exec \"\$\{CMD\[@\]\}\"$", f"{PREPARE_LINE}\n{EXEC_LINE}", text,
                 'exec "${CMD[@]}" line', flags=re.M)


def weights_wiring_problems(text: str, *, nearest_model: str) -> list[str]:
    """What is wrong with a staged run.sh's weights settings; empty when the chips will load the
    weights in model-dir and nothing names the nearest model."""
    problems = []
    lines = text.splitlines()
    for var, want in (("HF_MODEL", WIRING_LINES[0]), ("MODEL_WEIGHTS_DIR", WIRING_LINES[1])):
        sets = [ln for ln in lines if re.match(rf"\s*(?:export\s+)?{var}=", ln)]
        if sets != [want]:
            problems.append(f"run.sh must set {var} exactly once, as {want!r}; found {sets}")
    cmd = [ln for ln in lines if ln.startswith("CMD=(")]
    if len(cmd) != 1 or cmd[0].count("--model ") != 1 or f"--model {MODEL_DIR} " not in cmd[0]:
        problems.append(f"the run.sh command must pass --model {MODEL_DIR} once")
    if re.search(r"--(?:tokenizer-)?revision\b", text):
        problems.append("run.sh pins a --revision, which a local model-dir does not have")
    if PREPARE_LINE not in lines or EXEC_LINE not in lines or lines.index(PREPARE_LINE) > lines.index(EXEC_LINE):
        problems.append("run.sh must run prepare_model_dir.py before it starts vLLM")
    if nearest_model in text:
        problems.append(f"run.sh names the nearest model {nearest_model}")
    return problems
