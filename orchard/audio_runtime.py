# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The audio runtime the harness provides for Stable Audio 3 models (design docs/superpowers/specs/2026-10-28-stable-audio-runtime-design.md).

A bundle or a container package serves most models. For Stable Audio 3 small music nothing does, so the harness carries its own path: the library's
own code on the host, with the diffusion transformer on one chip (orchard/skills/audio-chip-templates/chip_dit.py). This module answers three questions:

  * does the runtime implement this model? (`runtime_problems`: the template's own `config_problems` on the model's `model_config.json`),
  * are the interpreters and the library checkout it needs on this machine? (`environment_problems`),
  * what did the run decide? (`selected`: the ledger decision `AUDIO_RUNTIME_SELECTED`, written once before stage 1, so a resume keeps
    the choice the way it keeps the package format).

It opens no device and imports no ttnn.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path

from orchard import safeio

RUNTIME = "stable-audio-3-dit-on-chip"
REFERENCE_FILE = "audio_reference.json"      # stage 1's record
RESULT_FILE = "audio_result.json"            # stage 2's record
STAGE_DISK_GB = 3.0                          # free disk the audio stages need: the runs hold about 0.3 GB of arrays and the clips are small
AUDIO_RUNTIME_SELECTED = "audio runtime selected"
TEMPLATES = Path(__file__).with_name("skills") / "audio-chip-templates"
CHIP_DIT = TEMPLATES / "chip_dit.py"
CHIP_DECODER = TEMPLATES / "chip_decoder.py"
CONFIG_MAX_BYTES = 1 << 20                  # the same bound the stage 0 detector uses for model_config.json
SCRIPTS = ("chip_dit.py", "chip_decoder.py", "audio_runtimes.py", "chip_generate.py", "audio_chip_test.py")
REFERENCE_SCRIPT = Path(__file__).with_name("skills") / "audio-reference-templates" / "make_reference.py"

_chip_dit = None
_chip_decoder = None


def chip_dit():
    """The template module (loaded by path, once). It imports nothing heavy at load time."""
    global _chip_dit
    if _chip_dit is None:
        spec = importlib.util.spec_from_file_location("orchard_chip_dit", CHIP_DIT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _chip_dit = mod
    return _chip_dit


def chip_decoder():
    """The decoder template module (loaded by path, once). It imports nothing heavy at load time."""
    global _chip_decoder
    if _chip_decoder is None:
        spec = importlib.util.spec_from_file_location("orchard_chip_decoder", CHIP_DECODER)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _chip_decoder = mod
    return _chip_decoder


def decoder_problems(model_dir) -> list[str]:
    """Why the autoencoder decoder of the model in `model_dir` cannot run on the chip with the harness's decoder template, or [] when it can. The
    decoder is optional: a model whose decoder is not supported still runs with the decoder on the host."""
    path = (Path(model_dir) / "model_config.json").resolve()
    try:
        config = safeio.read_json(path, max_bytes=CONFIG_MAX_BYTES, expect=dict)
    except (OSError, ValueError, safeio.SafeReadError) as exc:
        return [f"model_config.json cannot be read: {exc}"]
    return chip_decoder().config_problems(config)


def runtime_problems(model_dir) -> list[str]:
    """Why the runtime cannot serve the model in `model_dir` (a snapshot directory), or [] when it can. A config that cannot be read is
    a reason, never a crash."""
    # A Hugging Face snapshot holds each file as a link into the cache's blobs directory, so the link is resolved here before the bounded read.
    path = (Path(model_dir) / "model_config.json").resolve()
    try:
        config = safeio.read_json(path, max_bytes=CONFIG_MAX_BYTES, expect=dict)
    except (OSError, ValueError, safeio.SafeReadError) as exc:
        return [f"model_config.json cannot be read: {exc}"]
    return chip_dit().config_problems(config)


@dataclass(frozen=True)
class AudioPaths:
    """Where the runtime's three programs live. `reference_python` has CPU torch and the library; `chip_python` has ttnn (the shared Tenstorrent
    environment, which also has torch and the library's dependencies; the library itself is found through PYTHONPATH=`src_dir`)."""
    reference_python: str
    chip_python: str
    src_dir: str

    def record(self) -> dict:
        return {"reference_python": self.reference_python, "chip_python": self.chip_python, "src_dir": self.src_dir}

    @classmethod
    def from_record(cls, rec: dict) -> "AudioPaths":
        return cls(rec["reference_python"], rec["chip_python"], rec["src_dir"])


def audio_paths(operator_home, environ=None) -> AudioPaths:
    """The default locations (`tt-orchard audio-reference-env` builds the first two), each replaceable by an environment variable."""
    env = os.environ if environ is None else environ
    home = Path(operator_home)
    return AudioPaths(
        reference_python=env.get("ORCHARD_AUDIO_PYTHON") or str(home / "orchard-venvs" / "stable-audio" / "bin" / "python"),
        chip_python=env.get("ORCHARD_TTNN_PYTHON") or str(home / ".tenstorrent-venv" / "bin" / "python"),
        src_dir=env.get("ORCHARD_AUDIO_SRC") or str(home / "orchard-src" / "stable-audio-3"))


def environment_problems(paths: AudioPaths) -> list[str]:
    """What is missing on this machine, each with the command that makes it. Only existence is checked here; the scripts check versions."""
    out = []
    if not os.access(paths.reference_python, os.X_OK):
        out.append(f"the reference interpreter {paths.reference_python} is missing; `tt-orchard audio-reference-env` builds it")
    if not os.access(paths.chip_python, os.X_OK):
        out.append(f"the Tenstorrent interpreter {paths.chip_python} is missing")
    if not (Path(paths.src_dir) / "stable_audio_3" / "model.py").is_file():
        out.append(f"the library checkout {paths.src_dir} is missing; `tt-orchard audio-reference-env` builds it")
    return out


def script_hashes() -> dict:
    """sha256 of each program the runtime runs, recorded with the selection so a later change of the files is visible."""
    return {name: hashlib.sha256((TEMPLATES / name).read_bytes()).hexdigest() for name in SCRIPTS if (TEMPLATES / name).is_file()}


def selected(entries: list[dict]) -> dict | None:
    """The run's runtime selection (the last `AUDIO_RUNTIME_SELECTED` decision), or None."""
    found = None
    for e in entries:
        if e["event"] == "decision" and e["data"].get("decision") == AUDIO_RUNTIME_SELECTED:
            found = e["data"]
    return found


def selection_from_run_dir(run_dir) -> AudioPaths | None:
    """The paths of the run's selection, read from its ledger file (a gate gets only directories). None when the ledger has none or cannot be read."""
    import json
    entries = []
    try:
        with open(Path(run_dir) / "ledger.jsonl", encoding="utf-8") as f:
            for line in f:
                if AUDIO_RUNTIME_SELECTED in line:
                    try:
                        entries.append(json.loads(line))
                    except ValueError:
                        continue
    except OSError:
        return None
    return selection_paths(entries)


def selection_paths(entries: list[dict]) -> AudioPaths | None:
    rec = selected(entries)
    try:
        return AudioPaths.from_record(rec["paths"])        # a missing record or paths, or a paths value that is not an object, raises below
    except (KeyError, TypeError):
        return None
