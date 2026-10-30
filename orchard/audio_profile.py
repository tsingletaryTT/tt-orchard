# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Per-model audio profiles: the facts about an audio model that the audio stages and the package builder used to assume were Stable Audio's (issue 263).

A profile states, for one model family:
  sample_rate     the rate of the decoded waveform (Stable Audio 44.1 kHz, ACE-Step 48 kHz);
  clip_name       the listenable file a run writes under `clips/` (`audio.flac` or `audio.wav`);
  clip_equals_waveform  whether that file is the recorded waveform (ACE-Step's pipeline peak-normalises the clip after the array is recorded);
  required        the recorded arrays (kind, name) a run of this model must hold, as a function of its step count;
  stage2          which agreement checks apply: "end_to_end" compares every recorded array of the chip run with the CPU run (a short sampler or a
                  model whose chip stage keeps the loop on the CPU); "per_step" feeds the CPU run's recorded input of each step to the chip stage
                  and compares the velocities (`orchard.step_bars`), and decodes the CPU run's final latent on the chip; a sampler loop that feeds
                  its own errors back needs the second;
  runtime_files   the files a package of this model copies as its runtime (source relative to the repository, destination relative to the package);
  settings        the recorded settings of the standard run.
This module imports nothing outside the standard library, so the supervisor and the stage table can read it. `layout_problems` checks a run's manifest
against a profile; `sound_problems` (numpy) checks the waveform at the profile's rate. A model with no profile has no stage 2 here: `for_model` returns None
and the caller must say so.

What it does not do: wire the profiles into `audio_plan.reference_plan` and the supervisor's stage commands (those still name the Stable Audio library);
that is left as an issue. The profile is the data those changes will read.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

STAGE2_END_TO_END = "end_to_end"
STAGE2_PER_STEP = "per_step"


@dataclass(frozen=True)
class AudioProfile:
    key: str
    model_ids: tuple
    sample_rate: int
    clip_name: str
    clip_equals_waveform: bool             # False when the pipeline normalises the clip after the waveform array is recorded
    channels: int
    stage2: tuple
    steps: int | None                      # the standard run's sampler steps (None: read from the run's settings)
    latent_channels: int | None            # [channels, frames] of the final latent, or None when the run does not record it
    latent_rate_hz: float | None           # latent frames per second of audio, or None
    runtime_files: tuple = ()
    required_fixed: tuple = ()             # (kind, name) pairs every run holds
    per_step_kinds: tuple = ()             # (kind, name template) pairs held once per step; {k} is the step number
    settings: dict = field(default_factory=dict)

    def required(self, steps: int | None = None) -> list[tuple]:
        n = self.steps if steps is None else steps
        out = list(self.required_fixed)
        if self.per_step_kinds:
            if n is None:
                raise ValueError(f"{self.key}: the step count is needed to list the per-step arrays")
            out += [(kind, tmpl.format(k=k)) for k in range(n) for kind, tmpl in self.per_step_kinds]
        return out


STABLE_AUDIO = AudioProfile(
    key="stable-audio", model_ids=("stabilityai/stable-audio-3-small-music", "stabilityai/stable-audio-3-small-sfx", "stabilityai/stable-audio-3-medium"),
    sample_rate=44100, clip_name="audio.flac", clip_equals_waveform=True, channels=2, stage2=(STAGE2_END_TO_END,), steps=None, latent_channels=None, latent_rate_hz=None,
    required_fixed=(("waveform", "audio"),),
    settings={"sample_size": 5292032})

ACESTEP_TURBO = AudioProfile(
    key="acestep-1.5-turbo", model_ids=("ACE-Step/Ace-Step1.5",),
    sample_rate=48000, clip_name="audio.wav", clip_equals_waveform=False, channels=2, stage2=(STAGE2_PER_STEP,), steps=8, latent_channels=64, latent_rate_hz=25.0,
    runtime_files=(("scripts/acestep/acestep_run.py", "runtime/acestep_run.py"), ("scripts/acestep/acestep_report.py", "runtime/acestep_report.py")),
    required_fixed=(("noise", "x0"), ("conditioning", "encoder_states"), ("conditioning", "context"), ("latent", "final"), ("waveform", "audio")),
    per_step_kinds=(("step_latent", "x{k}"), ("step_denoised", "v{k}")),
    settings={"steps": 8, "cfg_scale": 1.0, "sampler": "turbo-ode-shift1", "duration": 30.0})

PROFILES = {p.key: p for p in (STABLE_AUDIO, ACESTEP_TURBO)}


def for_model(model_id) -> AudioProfile | None:
    """The profile of a repository id (a revision after `@` is ignored), or None for a model with no profile."""
    repo = str(model_id or "").split("@", 1)[0]
    for p in PROFILES.values():
        if repo in p.model_ids:
            return p
    return None


def layout_problems(manifest: dict, profile: AudioProfile) -> list[str]:
    """Reasons a recorded run (its parsed manifest, as `audio_reference` writes it) does not have the layout this profile expects: missing arrays,
    a waveform of the wrong channel count or length for the rate and the requested duration, a final latent of the wrong shape."""
    arts = {(a.get("kind"), a.get("name")): a for a in manifest.get("artifacts", []) if isinstance(a, dict)}
    settings = manifest.get("settings") or {}
    steps = settings.get("steps", profile.steps)
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        return [f"settings.steps is {steps!r}; the profile needs a positive step count"]
    out = []
    if profile.steps is not None and steps != profile.steps:
        out.append(f"the run has {steps} steps; the {profile.key} profile is for {profile.steps}")
    try:
        need = profile.required(steps)
    except ValueError as exc:
        return [str(exc)]
    out += [f"missing array {kind}/{name}" for kind, name in need if (kind, name) not in arts]
    duration = settings.get("duration")
    wave = arts.get(("waveform", "audio"))
    if wave is not None and isinstance(duration, (int, float)) and not isinstance(duration, bool) and duration > 0:
        shape = wave.get("shape") or []
        want = duration * profile.sample_rate
        if len(shape) != 2 or shape[0] != profile.channels:
            out.append(f"waveform shape {shape} is not [{profile.channels}, samples]")
        elif abs(shape[1] - want) > max(1, 0.05 * want):
            out.append(f"waveform has {shape[1]} samples; {duration} s at {profile.sample_rate} Hz is {int(want)} (a different rate, or a different length)")
    latent = arts.get(("latent", "final"))
    if latent is not None and profile.latent_channels is not None and isinstance(duration, (int, float)) and duration > 0:
        shape = latent.get("shape") or []
        if len(shape) != 2 or shape[0] != profile.latent_channels:
            out.append(f"final latent shape {shape} is not [{profile.latent_channels}, frames]")
        elif profile.latent_rate_hz and abs(shape[1] - duration * profile.latent_rate_hz) > max(2, 0.05 * duration * profile.latent_rate_hz):
            out.append(f"final latent has {shape[1]} frames; {duration} s at {profile.latent_rate_hz} Hz is {duration * profile.latent_rate_hz:g}")
    sample_size = settings.get("sample_size")
    if isinstance(sample_size, int) and isinstance(duration, (int, float)) and duration > 0 and abs(sample_size - duration * profile.sample_rate) > profile.sample_rate * 0.05 * duration \
            and profile.key != "stable-audio":
        out.append(f"settings.sample_size {sample_size} does not match {duration} s at {profile.sample_rate} Hz")
    return out


def sound_problems(run_dir, profile: AudioProfile) -> list[str]:
    """The sound checks of `orchard.audio_stages`, run at the profile's sample rate on a run directory's recorded waveform."""
    import numpy as np
    from orchard import audio_reference as ar
    from orchard import audio_stages
    with ar.load_run(Path(run_dir)) as run:
        wave = np.asarray(run.array("waveform", "audio"))
        duration = float(run.settings.get("duration", 0))
    checks = audio_stages.sound_checks(wave, profile.sample_rate, duration)
    return [] if checks["ok"] else list(checks["reasons"])


def run_problems(run_dir, profile: AudioProfile) -> list[str]:
    """Layout and sound problems of one run directory, read through `audio_reference.load_run` (which verifies every array against its hash)."""
    from orchard import audio_reference as ar
    with ar.load_run(Path(run_dir)) as run:
        manifest = run.manifest
    return layout_problems(manifest, profile) + sound_problems(run_dir, profile)


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Check a recorded audio run against a model profile.")
    p.add_argument("run")
    p.add_argument("--profile", required=True, choices=sorted(PROFILES))
    a = p.parse_args(argv)
    problems = run_problems(a.run, PROFILES[a.profile])
    for x in problems:
        print("PROBLEM", x)
    print("profile check:", "FAIL" if problems else "pass")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
