# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The two resident runtimes of the Stable Audio 3 path, each keeping the contract of orchard/resident.py: `load()` once, `generate(settings, out_dir)`
many times, `close()`.

    LibraryRuntime   the library's own float32 `generate` on the CPU (the comparison).
    ChipRuntime      the same, with the diffusion transformer's 20-layer stack on one Tenstorrent chip (chip_dit.py). It imports ttnn, so run it under a
                     gozer lease.

`load()` does what a one-shot run did before its first generation: read the model, check the chip path implements its configuration, open the
device and upload the weights. `generate()` does one generation: it makes a recorder (the same arrays and manifest as the CPU reference), runs
the library's `generate` with the recording hooks, and writes the clip and `chip.json`. Nothing carries over from one call to the next except the
loaded model and the chip's weights and kernel caches. The noise comes from the library's seeded generator inside `generate`, so the same
settings give the same arrays on every call.

The recorded timings of a run are the timings a one-shot script recorded: `conditioning_s`, `generate_and_decode_s`, and on the chip
`chip_transformer_call_s` (the transformer's calls summed: host preparation, uploads, the layers on the chip and the read back, so an upper bound on
device time), `chip_passes`. A caller can add numbers through `extra_timings` (a one-shot script adds `model_load_s` and `chip_weights_upload_s`
to its one run). Each run also records `host_loadavg_1m`, the host's one-minute load average when the run ended.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from orchard import resident                                    # noqa: E402  (the repository root is on the path just above)


def load_sibling(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def settings_problems(s: dict) -> list[str]:
    """What is wrong with the settings of a generation, before any model is touched."""
    problems = []
    for key in ("prompt", "duration", "steps", "cfg_scale", "seed", "sample_size"):
        if key not in s:
            problems.append(f"setting {key!r} is missing")
    if problems:
        return problems
    if not isinstance(s["seed"], int) or isinstance(s["seed"], bool) or s["seed"] < 0:
        problems.append("seed must be a whole number, 0 or more")
    if not isinstance(s["steps"], int) or isinstance(s["steps"], bool) or s["steps"] < 1:
        problems.append("steps must be a whole number, at least 1")
    for key in ("duration", "cfg_scale"):
        v = s[key]
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
            problems.append(f"{key} must be a finite number")
    if not problems and s["duration"] <= 0:
        problems.append("duration must be above 0")
    if not isinstance(s["sample_size"], int) or isinstance(s["sample_size"], bool) or s["sample_size"] < 1:
        problems.append("sample_size must be a whole number, at least 1")
    return problems


class LibraryRuntime:
    """The library's own run on the CPU, float32, recorded in the CPU reference layout."""
    device_label = "cpu"
    dtype = "float32"

    def __init__(self, model: str, *, threads: int = 0, record_layers: bool = True):
        self.model_name, self.threads, self.record_layers = model, threads, record_layers
        self.model = self.config_path = self.ckpt_path = None
        self.mk = self.torch = self.Recorder = None
        self.closed = False
        self.t_start = time.time()
        self.load_timings: dict = {}

    # -- load ---------------------------------------------------------------------------------------------------------------------------
    def load(self) -> dict:
        if self.model is not None:
            raise resident.ResidentError("the runtime is already loaded")
        self.mk = load_sibling(REPO / "orchard" / "skills" / "audio-reference-templates" / "make_reference.py", "make_reference_for_runtime")
        self.mk.offline_environment()
        import torch
        from orchard.audio_reference import Recorder
        self.torch, self.Recorder = torch, Recorder
        if self.threads:
            torch.set_num_threads(self.threads)
        t0 = time.time()
        self.model, self.config_path, self.ckpt_path = self.mk.load_model(self.model_name)
        self.load_timings = {"model_load_s": round(time.time() - t0, 3)}
        self._after_model_load()
        return dict(self.load_timings)

    def _after_model_load(self) -> None:
        """The hook of a subclass that needs more before the first generation (the chip's config check, device and weights)."""

    # -- one generation -----------------------------------------------------------------------------------------------------------------
    def _record_layers(self, wanted: bool):
        """The callable that records layer outputs of the first pass on a runtime that makes its own (None: the library's hooks record them)."""
        return None

    def _run_settings(self, s: dict) -> dict:
        sampler = s.get("sampler") or ("pingpong" if self.model.model.diffusion_objective == "rf_denoiser" else "euler")
        return {"prompt": s["prompt"], "duration": s["duration"], "steps": s["steps"], "cfg_scale": s["cfg_scale"], "sampler": sampler,
                "seed": s["seed"], "dtype": self.dtype, "sample_size": s["sample_size"], "device": self.device_label}

    def _before_generate(self, record_layers: bool, recorder) -> None:
        """A subclass points its own recording at this generation's recorder."""

    def _after_generate(self, recorder) -> None:
        """A subclass adds its own timings to the recorder."""

    def generate(self, settings: dict, out_dir=None, extra_timings=None) -> dict:
        """One generation. With no `out_dir` the run is recorded into a directory that is removed and no clip is written (a warm-up)."""
        if self.closed:
            raise resident.ResidentError("the runtime is closed")
        if self.model is None:
            raise resident.ResidentError("load() has not been called")
        problems = settings_problems(settings)
        if problems:
            raise resident.ResidentError("; ".join(problems))
        mk, torch = self.mk, self.torch
        t_call = time.time()
        temp = None
        if out_dir is None:                                      # a run nobody keeps (a warm-up): record into a directory that is removed
            temp = tempfile.mkdtemp(prefix="audio-runtime-")
            out = Path(temp) / "run"
        else:
            out = Path(out_dir)
        try:
            run_settings = self._run_settings(settings)
            recorder = self.Recorder(out, model={"name": self.model_name, "revision": self.ckpt_path.parent.name,
                                                 "config_sha256": mk.file_facts(self.config_path)["sha256"],
                                                 "diffusion_objective": str(self.model.model.diffusion_objective),
                                                 "checkpoint_bytes": mk.file_facts(self.ckpt_path, hash_it=False)["bytes"]},
                                     library={**mk.library_commit(), "source_verified": False}, settings=run_settings,
                                     seeds={"torch": settings["seed"], "numpy": None})
            self._before_generate(self.record_layers, recorder)
            mk.run_reference(self.model, run_settings, recorder, torch, layers=None if self.device_label == "cpu" else [])
            self._after_generate(recorder)
            recorder.timings.update({k: v for k, v in (extra_timings or {}).items()})
            load = resident.host_loadavg_1m()
            if load is not None:
                recorder.timings["host_loadavg_1m"] = load
            timings = {k: v for k, v in recorder.timings.items() if k not in (extra_timings or {}) and k != "host_loadavg_1m"}
            recorder.finish()
            sha = None
            if temp is None:
                sha = self._write_clip(out, run_settings, recorder, t_call)
            return {"timings": timings, "output_sha256": sha, "dir": str(out) if temp is None else None}
        finally:
            if temp is not None:
                shutil.rmtree(temp, ignore_errors=True)

    def _write_clip(self, out: Path, run_settings: dict, recorder, t_call: float) -> str | None:
        """The clip from the recorded waveform and `chip.json`; the hash of the waveform array's bytes. A failure to write the clip is printed and
        the run stays valid."""
        sha = None
        try:
            import numpy as np
            import soundfile
            from orchard.audio_reference import load_run
            with load_run(out) as run:
                wave = np.ascontiguousarray(np.asarray(run.array("waveform", "audio")))
            sha = hashlib.sha256(wave.tobytes()).hexdigest()
            clips = out / "clips"
            clips.mkdir(exist_ok=True)
            soundfile.write(str(clips / "audio.flac"), wave.T, 44100)
        except Exception as exc:                                  # noqa: BLE001
            print(f"clip not written: {exc!r}", file=sys.stderr)
        (out / "chip.json").write_text(json.dumps({"timings": recorder.timings, "settings": run_settings, "device": self.device_description(),
                                                   "total_s": round(time.time() - self.t_start, 3), "run_s": round(time.time() - t_call, 3)},
                                                  indent=1), encoding="utf-8")
        return sha

    def device_description(self) -> str:
        return "cpu, float32"

    def close(self) -> None:
        self.closed = True


class ChipRuntime(LibraryRuntime):
    """The library's run with the diffusion transformer's stack on one chip."""
    device_label = "tt"

    def __init__(self, model: str, *, device_id: int = 0, weights_dtype: str = "bfloat16", threads: int = 0, record_layers: bool = True):
        super().__init__(model, threads=threads, record_layers=record_layers)
        self.device_id, self.dtype = device_id, weights_dtype
        self.ttnn = self.device = self.chip = self.chip_dit = None
        self._last_forward = 0.0
        self._last_passes = 0

    def _after_model_load(self) -> None:
        chip_dit_cfg = load_sibling(HERE / "chip_dit.py", "chip_dit_config")
        problems = chip_dit_cfg.config_problems(json.loads(Path(self.config_path).read_text(encoding="utf-8")))
        if problems:
            raise SystemExit("the chip path does not implement this model's configuration:\n  " + "\n  ".join(problems))
        import ttnn                                    # after the lease; this opens the runtime
        self.ttnn = ttnn
        self.chip_dit = load_sibling(HERE / "chip_dit.py", "chip_dit_runtime")
        self.device = ttnn.open_device(device_id=self.device_id)
        dit = self.model.model.model.model             # StableAudioModel.model (conditioned wrapper) -> .model (DiTWrapper) -> .model (network)
        self.chip = self.chip_dit.ChipTransformer(ttnn, self.torch, self.device, dit.transformer, record=None, dtype=getattr(ttnn, self.dtype))
        self.chip_dit.install(dit, self.chip)
        self.load_timings["chip_weights_upload_s"] = self.chip.timings["upload_weights_s"]

    def _before_generate(self, record_layers: bool, recorder) -> None:
        chip, mk = self.chip, self.mk
        chip.record = (lambda name, t: recorder.record("layer", name, mk._numpy(t))) if record_layers else None
        chip.record_enabled = record_layers              # the first pass of each generation is recorded again
        chip._pending = {}
        chip.timings["forward_s"], chip.timings["passes"] = 0.0, 0

    def _after_generate(self, recorder) -> None:
        recorder.timings.update({"chip_transformer_call_s": round(self.chip.timings["forward_s"], 3), "chip_passes": self.chip.timings["passes"]})

    def device_description(self) -> str:
        return "blackhole, one chip, device id %d" % self.device_id

    def close(self) -> None:
        if self.ttnn is not None and self.device is not None and not self.closed:
            self.ttnn.close_device(self.device)
        super().close()
