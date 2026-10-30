# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The audio stages of a run whose model the harness's own audio runtime serves (orchard/audio_runtime.py).

Stage table for such a run (orchard/stages.py `spec_for`):

    0  triage                  as for any model (the template's audio branch)
    1  CPU reference           supervisor code, no hardware: the library's own `generate` on the CPU, for the model and for its base (the control),
                               plus one more run of the model in the interpreter the chip run uses (to measure what a change of CPU environment
                               alone does to the numbers)
    2  clip on one chip        supervisor code with one hardware test (orchard/skills/audio-chip-templates): the diffusion transformer on the
                               chip for the model and for its base, then the comparison with stage 1's runs
    3..7                       skipped, each with a reason
    8  operator bundle         the bundle builder's audio section reads `stages/2/audio_result.json`

Everything a gate relies on is re-derived from the saved arrays and clips. The result files hold the numbers for people to read and for the bundle;
a gate recomputes each one and refuses a file that disagrees.

Bars. The agreement bars are the provisional ones in `audio_reference.DEFAULT_BARS` unless stage 1's directory holds a usable noise-floor record
(`noise_floor.json`, orchard/audio_bars.py); then the measured bars decide. The result says which, and says "no floor measured" when there is none. The sound checks (duration, level,
clipping) use the audio spec's numbers (docs/superpowers/specs/2026-10-15-audio-bringup-design.md, section 6), which were set from 24 real clips.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np

from orchard import audio_bars
from orchard import audio_reference as ar
from orchard import resident
from orchard import safeio
from orchard.audio_runtime import REFERENCE_FILE, RESULT_FILE, STAGE_DISK_GB
from orchard.audio_plan import (CONFIG_FILE, DURATION_S, LIBRARY_NAMES, MODEL_SETTINGS, PROMPT, ROLE_CHIP, ROLE_CONTROL_CHIP, ROLE_CONTROL_REFERENCE,
                                ROLE_HOST, ROLE_REFERENCE, SAMPLE_RATE, clip_name_of, sample_rate_of, SAMPLE_SIZE, SCHEMA, SEED, AudioStageError, agree, chip_plan,
                                chip_test_command, inside, library_name, load_json, reference_command, reference_plan, settings_for,
                                repeat_name, snapshot_identity, write_chip_config, write_json)

# Sound checks. Source: the audio design, section 6 (24 clips measured on 2026-10-15).
DURATION_TOLERANCE = 0.05
LOUD_WINDOW_MS = 100
LOUD_DBFS = -40.0
MIN_LOUD_FRACTION = 0.40
MAX_FULL_SCALE_FRACTION = 0.01
FULL_SCALE = 0.9999



# ---- the sound checks ---------------------------------------------------------------------------------------------------

def sound_checks(wave, sample_rate: int, requested_s: float) -> dict:
    """Duration, sample rate, channels, the share of 100 ms windows louder than -40 dBFS (RMS over channels) and the share of samples at full
    scale, for a waveform of shape [channels, samples]. `ok` is true when every check meets its bar. Non-finite samples fail."""
    w = np.asarray(wave, dtype=np.float64)
    if w.ndim != 2 or w.shape[1] == 0:
        return {"ok": False, "reasons": ["the waveform is not [channels, samples] or is empty"]}
    if not np.isfinite(w).all():
        return {"ok": False, "reasons": ["the waveform has a value that is not finite"]}
    duration = w.shape[1] / sample_rate
    win = max(1, int(sample_rate * LOUD_WINDOW_MS / 1000))
    n = w.shape[1] // win
    rms = np.sqrt((w[:, :n * win].reshape(w.shape[0], n, win) ** 2).mean(axis=(0, 2))) if n else np.zeros(0)
    loud = float((20 * np.log10(np.maximum(rms, 1e-12)) > LOUD_DBFS).mean()) if n else 0.0
    full = float((np.abs(w) >= FULL_SCALE).mean())
    reasons = []
    if abs(duration - requested_s) > DURATION_TOLERANCE * requested_s:
        reasons.append(f"duration {duration:.2f} s is more than {DURATION_TOLERANCE:.0%} from the requested {requested_s} s")
    if loud < MIN_LOUD_FRACTION:
        reasons.append(f"only {loud:.1%} of {LOUD_WINDOW_MS} ms windows are above {LOUD_DBFS} dBFS; the bar is {MIN_LOUD_FRACTION:.0%}")
    if full > MAX_FULL_SCALE_FRACTION:
        reasons.append(f"{full:.2%} of samples are at full scale; the bar is {MAX_FULL_SCALE_FRACTION:.0%}")
    return {"ok": not reasons, "reasons": reasons, "duration_s": round(duration, 3), "sample_rate": int(sample_rate), "channels": int(w.shape[0]),
            "loud_window_fraction": round(loud, 4), "full_scale_fraction": round(full, 6)}


def write_clip(run_dir, sample_rate: int = SAMPLE_RATE, clip_name: str = "audio.flac") -> Path:
    """`clips/<clip_name>` of a recorded run, written from its waveform array (a run that already has one keeps it). The array's bytes are
    re-hashed by `load_run`, so the clip is the recorded waveform."""
    import soundfile
    clip = Path(run_dir) / "clips" / clip_name
    if clip.is_file():
        return clip
    with ar.load_run(run_dir) as run:
        wave = np.asarray(run.array("waveform", "audio"))
    clip.parent.mkdir(exist_ok=True)
    soundfile.write(str(clip), wave.T, sample_rate)
    return clip


def clip_matches_waveform(run_dir, wave, sample_rate: int = SAMPLE_RATE, clip_name: str = "audio.flac", exact: bool = True) -> list[str]:
    """Reasons the clip file is not the recorded waveform: unreadable, or another length, rate or channel count. The FLAC is lossless on 16-bit
    samples, so the decoded values are compared with the waveform to within one 16-bit step. With `exact` false (a model whose pipeline normalises
    the loudness of the clip after the waveform is recorded, see `audio_profile.AudioProfile.clip_equals_waveform`) only the file, its rate and its
    shape are checked."""
    clip = Path(run_dir) / "clips" / clip_name
    try:
        import soundfile
        data, rate = soundfile.read(str(clip), always_2d=True, dtype="float64")
    except Exception as exc:                                    # a missing file, a bad file or no soundfile
        return [f"the clip {clip.name} cannot be read: {type(exc).__name__}"]
    w = np.asarray(wave, dtype=np.float64)
    if rate != sample_rate or data.shape != w.T.shape:
        return [f"the clip is {data.shape[1]} channels, {data.shape[0]} samples at {rate} Hz; the waveform is {w.shape[0]} channels, {w.shape[1]} samples"]
    if exact and float(np.abs(data - w.T).max()) > 2.0 / 32768:
        return ["the clip's samples differ from the recorded waveform by more than one 16-bit step"]
    if not exact:
        # A normalised clip is the recorded waveform times a gain (peak normalisation is linear), so it must correlate with it almost perfectly; a silent
        # or unrelated file with the same rate and shape does not.
        x, y = data.ravel(), w.T.ravel()
        if x.std() == 0 or y.std() == 0 or float(np.corrcoef(x, y)[0, 1]) < 0.999:
            return ["the clip is not the recorded waveform up to a gain (correlation below 0.999, or silent)"]
    return []


# ---- comparisons --------------------------------------------------------------------------------------------------------

def summarise(report: dict) -> dict:
    """The numbers of a `compare_runs` report that people read and the bundle prints: the verdict, each failing array, the final latent, the
    waveform (with its log-spectrum distance) and the worst layer."""
    results = report.get("results", [])
    by = {(r["kind"], r["name"]): r for r in results}

    def pick(kind, name, extra=()):
        r = by.get((kind, name))
        if r is None or not r.get("comparable"):
            return None
        return {k: r[k] for k in ("pcc", "rel_l2", "slope", *extra) if k in r}

    layers = [r for r in results if r["kind"] == "layer" and r.get("comparable")]
    worst = max(layers, key=lambda r: r["rel_l2"], default=None)
    return {"ok": bool(report.get("ok")), "comparable": bool(report.get("comparable")), "problems": list(report.get("problems", [])),
            "arrays": len(results), "failed": [f"{r['kind']}/{r['name']}" for r in results if not r.get("ok")],
            "latent_final": pick("latent", "final"), "waveform": pick("waveform", "audio"),
            "worst_layer": None if worst is None else {"name": worst["name"], "rel_l2": worst["rel_l2"], "pcc": worst["pcc"]},
            "bars": "provisional: audio_reference.DEFAULT_BARS (a choice, not yet set from a measured noise floor)"}


def waveform_distance(ref_run, cand_run) -> float | None:
    """The log-spectrum distance of the two waveforms (audio_reference.log_spectrum_distance), or None."""
    return ar.log_spectrum_distance(ref_run.array("waveform", "audio"), cand_run.array("waveform", "audio"))


def compare(ref_dir, cand_dir, *, informational: bool = False, floor=None) -> dict:
    """`summarise(compare_runs(...))` of two run directories, float32 reference against the candidate with the dtype setting left out of the
    match, and the waveform's log-spectrum distance added. `floor` (an `audio_bars.Floor`) supplies measured bars and the text that says so; without
    one the bars are the provisional defaults. `informational` says the numbers are reported and no bar decides anything (the base
    control: fifty guided steps compound the same per-pass error the fine-tune's eight steps show)."""
    with ar.load_run(ref_dir) as ref, ar.load_run(cand_dir) as cand:
        out = summarise(ar.compare_runs(ref, cand, None if floor is None else floor.bars, cross_precision=True))
        if floor is not None:
            out["bars"] = floor.describe({k for k, _ in ref.entries})
        if informational and floor is not None and floor.measured:
            out["bars"] = "informational: the bars from the measured floor are shown as met or failed, and the control's verdict rests on the discrimination check"
        elif informational:
            out["bars"] = "informational: the provisional bars are shown as met or failed, and the control's verdict rests on the discrimination check"
        if out["comparable"] and out["waveform"] is not None:
            out["waveform"]["log_spectrum_distance"] = waveform_distance(ref, cand)
    return out


def control_difference(fine_dir, base_dir) -> dict:
    """The base's waveform must differ from the fine-tune's (`audio_reference.compare_control`): a run that served the wrong weights would not."""
    with ar.load_run(fine_dir) as ft, ar.load_run(base_dir) as base:
        r = ar.compare_control(ft, base, require_matched=False)
    return {"ok": bool(r.get("ok")), "rel_l2": r.get("rel_l2"), "pcc": r.get("pcc"), "problems": list(r.get("problems", []))}


def closeness(ref_dir, cand_dir) -> float | None:
    """The relative L2 error of the candidate's final latent against the reference's."""
    with ar.load_run(ref_dir) as ref, ar.load_run(cand_dir) as cand:
        m = ar.metrics(np.asarray(ref.array("latent", "final")), np.asarray(cand.array("latent", "final")))
    return m.get("rel_l2") if m.get("comparable") else None


def control_discrimination(base_ref_dir, fine_ref_dir, chip_base_dir) -> dict:
    """The base's chip run must be nearer to the base's own CPU reference than to the fine-tune's. A path that served the wrong weights would
    be nearer to the other one. Both are relative L2 errors of the final latent."""
    own, other = closeness(base_ref_dir, chip_base_dir), closeness(fine_ref_dir, chip_base_dir)
    return {"ok": own is not None and other is not None and own < other, "own_rel_l2": own, "other_rel_l2": other}


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run_facts(run_dir, rel: str) -> dict:
    """What a result file records about one recorded run: where it is, its model revision and library commit, its settings, its timings and the
    sha256 of its manifest. `rel` is the run directory relative to the supervisor's run directory."""
    d = Path(run_dir) / rel
    with ar.load_run(d) as run:
        m = run.manifest
        return {"dir": rel, "model": m["model"]["name"], "revision": m["model"]["revision"], "config_sha256": m["model"].get("config_sha256"),
                "library_commit": m["library"]["commit"],
                "settings": dict(m["settings"]), "timings": dict(m.get("timings", {})),
                "manifest_sha256": sha256_file(d / "manifest.json")}


# ---- the files a stage writes and its gate re-derives ----------------------------------------------------------------------------

def identity_problems(facts: dict, expected: dict | None, library_commit: str | None, label: str) -> list[str]:
    """Reasons a recorded run is not a run of the model this stage was started on: another snapshot revision, another `model_config.json`, or another
    library commit. `expected` is {"revision", "config_sha256"} taken from the run's own snapshot (audio_plan.snapshot_identity); the library
    registry name alone does not say which snapshot a script loaded. A missing expectation is a reason, not a pass."""
    if not isinstance(expected, dict) or not expected.get("revision") or not expected.get("config_sha256"):
        return [f"{label}: no expected model revision and configuration were recorded, so nothing shows which snapshot the run loaded"]
    out = []
    if facts.get("revision") != expected["revision"]:
        out.append(f"{label}: it loaded revision {facts.get('revision')!r}; this run's snapshot is {expected['revision']!r}")
    if facts.get("config_sha256") != expected["config_sha256"]:
        out.append(f"{label}: its model_config.json is not this run's snapshot's")
    if library_commit and facts.get("library_commit") != library_commit:
        out.append(f"{label}: it ran library commit {facts.get('library_commit')!r}; the pinned commit is {library_commit!r}")
    return out


def check_run_dir(run_dir, entry: dict, want_settings: dict | None, label: str, problems: list, *, expected: dict | None = None,
                  library_commit: str | None = None, profile=None) -> dict | None:
    """Verify one recorded run named by a result entry: it is inside the run directory, its arrays match their hashes, its manifest hash is the
    recorded one, its settings are the expected ones and its waveform passes the sound checks (and the clip is that waveform). Returns the
    sound-check record, or None after adding to `problems`. `profile` (an `orchard.audio_profile.AudioProfile`) sets the sample rate, the clip file and
    whether the clip must equal the waveform; without one the Stable Audio values apply. The profile's recorded layout is checked as well."""
    rate, clip_name, exact = (SAMPLE_RATE, "audio.flac", True) if profile is None else (profile.sample_rate, profile.clip_name, profile.clip_equals_waveform)
    rel = entry.get("dir") if isinstance(entry, dict) else None
    p = inside(run_dir, rel)
    if p is None or not p.is_dir():
        problems.append(f"{label}: its directory {rel!r} is not a directory inside the run directory")
        return None
    try:
        facts = run_facts(run_dir, rel)
        with ar.load_run(p) as run:
            wave = np.asarray(run.array("waveform", "audio"))
    except (ar.ReferenceError, OSError, KeyError) as exc:
        problems.append(f"{label}: {exc}")
        return None
    if entry.get("manifest_sha256") != facts["manifest_sha256"]:
        problems.append(f"{label}: the manifest is not the one the result recorded")
    problems += identity_problems(facts, expected, library_commit, label)
    for key in ("model", "revision", "config_sha256", "library_commit", "settings", "timings"):     # copies in the record must be the manifest's
        if entry.get(key) != facts[key]:
            problems.append(f"{label}: its recorded {key} is not what its manifest says")
    if entry.get("clip") != f"{rel}/clips/{clip_name}":
        problems.append(f"{label}: its clip path is {entry.get('clip')!r}; the expected path is {rel}/clips/{clip_name}")
    if profile is not None:
        from orchard import audio_profile
        with ar.load_run(p) as run:
            problems += [f"{label}: {x}" for x in audio_profile.layout_problems(run.manifest, profile)]
    if want_settings is not None:
        got = {k: facts["settings"].get(k) for k in want_settings}
        if got != want_settings:
            problems.append(f"{label}: its settings are {got}; the expected settings are {want_settings}")
    checks = sound_checks(wave, rate, float(facts["settings"].get("duration", 0)))
    if not checks["ok"]:
        problems.append(f"{label}: the clip fails the sound checks: {'; '.join(checks['reasons'])}")
    problems += [f"{label}: {x}" for x in clip_matches_waveform(p, wave, rate, clip_name, exact)]
    return checks


# ---- stage 1: the CPU references -------------------------------------------------------------------------------------------------



def reference_problems(stage_dir, run_dir, expect_roles=(ROLE_REFERENCE, ROLE_HOST)) -> list[str]:
    """The gate of stage 1, from the files: every run in `audio_reference.json` verifies, has the expected settings and passes the sound checks;
    the same-environment floor is recomputed and equals the recorded one."""
    problems: list[str] = []
    try:
        rec = load_json(Path(stage_dir) / REFERENCE_FILE, REFERENCE_FILE)
    except AudioStageError as exc:
        return [str(exc)]
    runs = rec.get("runs")
    if not isinstance(runs, dict):
        return [f"{REFERENCE_FILE} needs a runs object"]
    for role in expect_roles:
        if role not in runs:
            problems.append(f"{REFERENCE_FILE} has no {role} run")
    sound = {}
    for role, entry in runs.items():
        name = entry.get("model") if isinstance(entry, dict) else None
        if name not in MODEL_SETTINGS:
            problems.append(f"{role}: model {name!r} is not one the audio runtime knows")
            continue
        sound[role] = check_run_dir(run_dir, entry, settings_for(name), role, problems,
                                    expected=(rec.get("expected") or {}).get(name) if isinstance(rec.get("expected"), dict) else None,
                                    library_commit=rec.get("library_commit"))
    if ROLE_REFERENCE in runs and ROLE_HOST in runs and not problems:
        a, b = inside(run_dir, runs[ROLE_REFERENCE].get("dir")), inside(run_dir, runs[ROLE_HOST].get("dir"))
        try:
            fresh = compare(a, b)
        except ar.ReferenceError as exc:
            return problems + [f"environment floor: {exc}"]
        if not agree(rec.get("environment_floor"), fresh):
            problems.append("the recorded environment floor is not what the two runs give now")
    return problems


def cached_run(cache_dir, name: str, expected: dict | None, library_commit: str | None):
    """A saved CPU reference of library model `name` that can stand in for a fresh one, or None: the directory loads and its arrays match their
    hashes, its model revision and library commit are the wanted ones and its settings are the stage's settings for that model. A reference is a
    deterministic float32 CPU run, so the same inputs give the same arrays (measured: a repeat run is bit-identical)."""
    if not cache_dir or not library_commit or not isinstance(expected, dict) or not expected.get("revision") or not expected.get("config_sha256"):
        return None
    revision = expected["revision"]
    d = Path(cache_dir) / name
    if not d.is_dir() or d.is_symlink():
        return None
    try:
        with ar.load_run(d) as run:
            m = run.manifest
            ok = (m["model"]["name"] == name and m["model"]["revision"] == revision and m["library"]["commit"] == library_commit
                  and m["model"].get("config_sha256") == expected["config_sha256"]
                  and {k: m["settings"].get(k) for k in settings_for(name)} == settings_for(name) and m["settings"].get("dtype") == "float32"
                  and m["settings"].get("device") == "cpu")
    except (ar.ReferenceError, OSError, KeyError, TypeError):
        return None
    return d if ok else None


def copy_cached_run(cache_dir, name: str, expected: dict | None, library_commit: str | None, out):
    """Validate the cache entry `name` (`cached_run`) and copy it to `out`, holding the entry's shared lock for both, so that a run replacing a stale entry
    (orchard/supervisor.py `save_reference`, which holds the exclusive lock) cannot change the directory between the check and the copy. Returns the entry's
    directory, or None when there is no valid entry or the copy failed; a partial copy is removed, and the caller then makes the reference fresh."""
    from orchard import cache_lock
    with cache_lock.entry_lock(cache_dir, name, exclusive=False):
        hit = cached_run(cache_dir, name, expected, library_commit)
        if hit is None:
            return None
        try:
            shutil.copytree(hit, out, symlinks=False)
        except (OSError, shutil.Error):
            shutil.rmtree(out, ignore_errors=True)
            return None
        return hit


def subprocess_runner(base_env: dict):
    """A runner for `run_references`: each command is a child in its own session with `base_env` plus the command's extra variables, its output in
    the log file; a timeout kills the whole session. Returns the exit status, None for a timeout."""
    import os
    import signal
    import subprocess

    def run(argv, env_extra, timeout_s, log_path):
        # The child runs in its own session so that a timeout here can kill everything it started. When this process is itself cancelled (the verb is
        # sent SIGTERM by orchard/audio_remote.py, which orchard/audio_cli.py turns into an exception), the `except BaseException` below kills that
        # session before the exception goes on, so no CPU run outlives its verb.
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "wb") as log:
            proc = subprocess.Popen(argv, env={**base_env, **env_extra}, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            try:
                return proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                return None
            except BaseException:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                raise
    return run


def run_references(stage_dir, run_dir, paths, plan, runner, *, expected: dict, library_commit: str | None, cache_dir=None,
                   script: Path | None = None, timeout_s: float = 3600.0, say=lambda text: None) -> dict:
    """Make every run of `plan` under `stage_dir` and write `audio_reference.json`. `runner(argv, env_extra, timeout_s, log_path)` runs one command
    and returns its exit status (None for a timeout); `expected` maps a library name to the identity of the snapshot the run was started on
    ({"revision", "config_sha256"}); every run is checked against it and against `library_commit` after it is made or copied, because the registry
    name alone does not say which snapshot a script loaded. Raises AudioStageError when a run fails or is another model. A reference (not the same-environment run) that the cache holds is copied instead of made again."""
    from orchard import audio_runtime
    script = script or audio_runtime.REFERENCE_SCRIPT
    stage_dir, run_dir = Path(stage_dir), Path(run_dir)
    rel_stage = stage_dir.resolve().relative_to(run_dir.resolve())
    (stage_dir / "log").mkdir(parents=True, exist_ok=True)
    runs = {}
    for item in plan:
        out = stage_dir / item["dir"]
        out.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        hit = (copy_cached_run(cache_dir, item["name"], expected.get(item["name"]), library_commit, out)
               if item["interpreter"] == "reference" else None)
        if hit is not None:
            say(f"{item['role']}: copied the cached reference {item['name']}")
            from_cache = True
        else:
            argv, env = reference_command(item, paths, rel_stage, script)
            argv[argv.index("--out") + 1] = str(out)
            say(f"{item['role']}: running {item['name']} on the CPU")
            code = runner(argv, env, timeout_s, stage_dir / "log" / f"{item['role']}.log")
            if code != 0:
                raise AudioStageError(f"the CPU run for {item['role']} ({item['name']}) {'timed out' if code is None else f'exited {code}'}; "
                                      f"see stages/{rel_stage.name}/log/{item['role']}.log")
            from_cache = False
        seconds = round(time.time() - t0, 3)
        try:
            write_clip(out, sample_rate_of(item), clip_name_of(item))
            facts = run_facts(run_dir, str(rel_stage / item["dir"]))
        except (ar.ReferenceError, OSError, KeyError) as exc:
            raise AudioStageError(f"the CPU run for {item['role']} cannot be read back: {exc}") from exc
        wrong = identity_problems(facts, expected.get(item["name"]), library_commit, item["role"])
        if wrong:
            raise AudioStageError("; ".join(wrong))
        runs[item["role"]] = {**facts, "role": item["role"], "interpreter": item["interpreter"], "from_cache": from_cache, "wall_seconds": seconds,
                              "clip": f"{facts['dir']}/clips/{clip_name_of(item)}"}
    rec = {"schema": SCHEMA, "runs": runs, "expected": expected, "library_commit": library_commit,
           "environment_floor": compare(Path(run_dir) / runs[ROLE_REFERENCE]["dir"], Path(run_dir) / runs[ROLE_HOST]["dir"]),
           "limits": ["The reference is the library's own float32 CPU run. The same-environment run is the same code in the interpreter that the chip "
                      "run uses; the difference between the two is what a change of CPU environment does to these numbers, before any chip is involved."]}
    write_json(stage_dir / REFERENCE_FILE, rec)
    return rec


# ---- stage 2: the clip on a chip ------------------------------------------------------------------------------------------------



def finish_chip(stage_dir, run_dir, ref_stage_dir, plan: list[dict], reference_roles: dict) -> dict:
    """After the hardware test: compare each chip run with its reference (and the fine-tune's with the same-environment run), check the control
    and the clips, and write `audio_result.json`. `reference_roles` maps a chip role to the reference role it is compared with. Raises
    AudioStageError when a recorded run is missing."""
    stage_dir, run_dir, ref_stage_dir = Path(stage_dir), Path(run_dir), Path(ref_stage_dir)
    ref_rec = load_json(ref_stage_dir / REFERENCE_FILE, REFERENCE_FILE)
    rel = stage_dir.resolve().relative_to(run_dir.resolve())
    floor = audio_bars.load_floor(ref_stage_dir)
    runs, agreement, sound = {}, {}, {}
    for item in plan:
        d = stage_dir / "chip" / item["name"]
        if not (d / "manifest.json").is_file():
            raise AudioStageError(f"the chip run for {item['role']} ({item['name']}) left no manifest")
        facts = run_facts(run_dir, str(rel / "chip" / item["name"]))
        wrong = identity_problems(facts, (ref_rec.get("expected") or {}).get(item["name"]) if isinstance(ref_rec.get("expected"), dict) else None,
                                  ref_rec.get("library_commit"), item["role"])
        if wrong:
            raise AudioStageError("; ".join(wrong))
        try:
            clip = write_clip(d, sample_rate_of(item), clip_name_of(item))
        except (ar.ReferenceError, OSError) as exc:
            raise AudioStageError(f"the chip clip for {item['name']} cannot be written: {exc}") from exc
        chip_json = {}
        try:
            chip_json = load_json(d / "chip.json", "chip.json")
        except AudioStageError:
            pass
        runs[item["role"]] = {**facts, "role": item["role"], "clip": f"{facts['dir']}/clips/{clip_name_of(item)}", "device": chip_json.get("device"),
                              "total_seconds": chip_json.get("total_s")}
        ref_role = reference_roles[item["role"]]
        ref_entry = ref_rec["runs"][ref_role]
        agreement[f"{item['role']}_vs_{ref_role}"] = compare(run_dir / ref_entry["dir"], d, informational=item["role"] == ROLE_CONTROL_CHIP, floor=floor)
    if ROLE_CHIP in runs and ROLE_HOST in ref_rec["runs"]:
        agreement["chip_vs_host_same_env"] = compare(run_dir / ref_rec["runs"][ROLE_HOST]["dir"], stage_dir / "chip" / plan[0]["name"])
    repeats = {}
    try:
        n_repeats = int(load_json(stage_dir / CONFIG_FILE, CONFIG_FILE).get("repeats", 1))
    except (AudioStageError, TypeError, ValueError):
        n_repeats = 1
    for item in plan:
        repeats[item["role"]] = [runs[item["role"]]]
        for k in range(1, n_repeats):
            rel_k = str(rel / "chip" / repeat_name(item["name"], k))
            if not (run_dir / rel_k / "manifest.json").is_file():
                raise AudioStageError(f"repeat {k} of the chip run for {item['role']} ({item['name']}) left no manifest")
            try:
                write_clip(run_dir / rel_k, sample_rate_of(item), clip_name_of(item))
            except (ar.ReferenceError, OSError) as exc:
                raise AudioStageError(f"the clip of repeat {k} for {item['role']} cannot be written: {exc}") from exc
            facts_k = run_facts(run_dir, rel_k)
            wrong = identity_problems(facts_k, (ref_rec.get("expected") or {}).get(item["name"]) if isinstance(ref_rec.get("expected"), dict) else None,
                                      ref_rec.get("library_commit"), f"{item['role']} repeat {k}")
            if wrong:
                raise AudioStageError("; ".join(wrong))
            repeats[item["role"]].append({**facts_k, "role": item["role"], "clip": f"{facts_k['dir']}/clips/{clip_name_of(item)}"})
    control = discrimination = None
    if ROLE_CONTROL_CHIP in runs:
        control = control_difference(stage_dir / "chip" / plan[0]["name"], stage_dir / "chip" / plan[1]["name"])
        discrimination = control_discrimination(run_dir / ref_rec["runs"][ROLE_CONTROL_REFERENCE]["dir"], run_dir / ref_rec["runs"][ROLE_REFERENCE]["dir"],
                                                stage_dir / "chip" / plan[1]["name"])
    result = {"schema": SCHEMA, "runtime": "stable-audio-3-dit-on-chip", "settings": settings_for(plan[0]["name"]), "runs": runs,
              "timing": timing_record(repeats, configured_warmup(stage_dir)),
              "reference_runs": ref_rec["runs"], "agreement": agreement, "environment_floor": ref_rec.get("environment_floor"),
              "noise_floor": floor.record(), "control_difference": control, "control_discrimination": discrimination, "repeats": repeats,
              "limits": ["Nobody has listened to the clips; no check here says the music is good.",
                         audio_bars.limit_line(floor),
                         ("The diffusion transformer and the autoencoder decoder's transformer ran on the chip. The T5Gemma encoder, the sampler, the decoder's first linear layer, "
                          "its final convolution and the waveform unpatching ran on the host." if chip_decoder_configured(stage_dir) else
                          "Only the diffusion transformer ran on the chip. The T5Gemma encoder, the sampler and the autoencoder decoder ran on the host."),
                         "Agreement is measured on one prompt, one duration and one seed."]}
    write_json(stage_dir / RESULT_FILE, result)
    return result


def chip_decoder_configured(stage_dir) -> bool:
    """True when the stage's config asked for the autoencoder decoder on the chip."""
    try:
        return load_json(Path(stage_dir) / CONFIG_FILE, CONFIG_FILE).get("chip_decoder") is True
    except AudioStageError:
        return False


def configured_warmup(stage_dir) -> int:
    """The number of discarded runs the stage's config asked for (0 when it says none or cannot be read)."""
    try:
        w = load_json(Path(stage_dir) / CONFIG_FILE, CONFIG_FILE).get("warmup", 0)
    except AudioStageError:
        return 0
    return w if isinstance(w, int) and not isinstance(w, bool) and w >= 0 else 0


def timing_record(repeats: dict, warmup: int) -> dict:
    """The timing record of a result: for each role, the label and the spread (min, median, max) of each timed part over its repeats, the load that
    the single loaded process reported once, and the host load average of each run (orchard/resident.py `summarise_runs`). `warmup_runs` is the number
    of discarded runs that came before the repeats."""
    return {"warmup_runs": warmup, "min_repeats": resident.MIN_REPEATS,
            "roles": {role: resident.summarise_runs([e.get("timings") for e in lst if isinstance(e, dict)], warmup) for role, lst in repeats.items()}}


def timing_problems(stage_dir, res: dict, repeats) -> list[str]:
    """The recorded timing is what the repeats give: the warm-up is the config's, each role has a record, its label is the label its number of repeats
    earns, its spread, load and load averages are recomputed from the repeats' timings, and the load is reported in repeat 0 only."""
    timing = res.get("timing")
    if not isinstance(timing, dict) or not isinstance(timing.get("roles"), dict):
        return [f"{RESULT_FILE} has no timing record"]
    problems = []
    if timing.get("warmup_runs") != configured_warmup(stage_dir):
        problems.append(f"timing: the recorded warm-up is {timing.get('warmup_runs')!r} runs; the config asked for {configured_warmup(stage_dir)}")
    if not isinstance(repeats, dict):
        return problems
    for role, lst in repeats.items():
        if not isinstance(lst, list):
            continue
        entries = [e for e in lst if isinstance(e, dict) and isinstance(e.get("timings"), dict)]
        for k, e in enumerate(entries):
            if k >= 1 and any(key in e["timings"] for key in resident.LOAD_KEYS):
                problems.append(f"{role} repeat {k}: it records a load time; the load happened once, in repeat 0")
        fresh = resident.summarise_runs([e["timings"] for e in entries], configured_warmup(stage_dir))
        rec = timing["roles"].get(role)
        if not isinstance(rec, dict):
            problems.append(f"{role}: the timing record has no entry for this role")
            continue
        if rec.get("label") != fresh["label"]:
            problems.append(f"{role}: the timing label is {rec.get('label')!r}; {fresh['repeats']} repeats earn {fresh['label']!r}")
        if not agree({k: v for k, v in rec.items() if k != "label"}, {k: v for k, v in fresh.items() if k != "label"}):
            problems.append(f"{role}: the recorded timing spread is not what the repeats give")
    return problems


def repeat_problems(stage_dir, run_dir, res: dict, ref: dict, runs: dict) -> list[str]:
    """The repeats listed in the result are the ones the config asked for, each loads (hashes match), is the expected model and has the recorded
    manifest, and the first of each list is the run itself."""
    problems: list[str] = []
    try:
        want = int(load_json(Path(stage_dir) / CONFIG_FILE, CONFIG_FILE).get("repeats", 1))
    except (AudioStageError, TypeError, ValueError):
        return [f"{CONFIG_FILE} cannot be read, so the repeats cannot be checked"]
    reps = res.get("repeats")
    if not isinstance(reps, dict):
        return [f"{RESULT_FILE} has no repeats record"]
    for role, run_entry in runs.items():
        lst = reps.get(role)
        if not isinstance(lst, list) or len(lst) != want:
            problems.append(f"{role}: the result lists {len(lst) if isinstance(lst, list) else 'no'} repeats; the config asked for {want}")
            continue
        for k, entry in enumerate(lst):
            if not isinstance(entry, dict):
                problems.append(f"{role} repeat {k}: not an object")
                continue
            name = run_entry.get("model") if isinstance(run_entry, dict) else None
            if name not in MODEL_SETTINGS:
                continue
            if entry.get("dir") != (run_entry.get("dir") if k == 0 else str(Path(run_entry["dir"]).with_name(repeat_name(name, k)))):
                problems.append(f"{role} repeat {k}: its directory is not the expected one")
                continue
            check_run_dir(run_dir, entry, settings_for(name), f"{role} repeat {k}", problems,
                          expected=(ref.get("expected") or {}).get(name) if isinstance(ref.get("expected"), dict) else None,
                          library_commit=ref.get("library_commit"))
    return problems


def chip_problems(stage_dir, run_dir, ref_stage_dir) -> list[str]:
    """The gate of stage 2, from the files. It recomputes every comparison and sound check and refuses a result that differs. A pass needs: the
    fine-tune's chip run agrees with its reference (provisional bars), its clip passes the sound checks, and (when the control ran) the base's chip
    run agrees with its reference and differs from the fine-tune's."""
    problems: list[str] = []
    try:
        res = load_json(Path(stage_dir) / RESULT_FILE, RESULT_FILE)
        ref = load_json(Path(ref_stage_dir) / REFERENCE_FILE, REFERENCE_FILE)
    except AudioStageError as exc:
        return [str(exc)]
    runs, ref_runs, agreement = res.get("runs"), ref.get("runs"), res.get("agreement")
    floor = audio_bars.load_floor(ref_stage_dir)
    if not agree(res.get("noise_floor"), floor.record()):
        problems.append(f"noise_floor: the result's record of its bars is not what {audio_bars.FLOOR_FILE} gives now ({floor.label[:120]})")
    if not isinstance(runs, dict) or not isinstance(ref_runs, dict) or not isinstance(agreement, dict):
        return [f"{RESULT_FILE} needs runs, reference_runs and agreement"]
    if ROLE_CHIP not in runs:
        return [f"{RESULT_FILE} has no chip run"]
    if ROLE_CONTROL_REFERENCE in ref_runs and ROLE_CONTROL_CHIP not in runs:
        problems.append(f"stage 1 made a control reference, so {RESULT_FILE} needs a {ROLE_CONTROL_CHIP} run, and has none")
    if not agree(res.get("reference_runs"), ref_runs):
        problems.append("the result's reference_runs are not stage 1's")
    for role, entry in runs.items():
        name = entry.get("model") if isinstance(entry, dict) else None
        if name not in MODEL_SETTINGS:
            problems.append(f"{role}: model {name!r} is not one the audio runtime knows")
            continue
        check_run_dir(run_dir, entry, {**settings_for(name)}, role, problems,
                      expected=(ref.get("expected") or {}).get(name) if isinstance(ref.get("expected"), dict) else None,
                      library_commit=ref.get("library_commit"))
    pairs = [(ROLE_CHIP, ROLE_REFERENCE)] + ([(ROLE_CONTROL_CHIP, ROLE_CONTROL_REFERENCE)] if ROLE_CONTROL_CHIP in runs else [])
    for chip_role, ref_role in pairs:
        key = f"{chip_role}_vs_{ref_role}"
        if ref_role not in ref_runs:
            problems.append(f"stage 1 made no {ref_role} run to compare {chip_role} with")
            continue
        rec = agreement.get(key)
        try:
            fresh = compare(Path(run_dir) / ref_runs[ref_role]["dir"], Path(run_dir) / runs[chip_role]["dir"], informational=chip_role == ROLE_CONTROL_CHIP,
                           floor=floor)
        except (ar.ReferenceError, KeyError) as exc:
            problems.append(f"{key}: {exc}")
            continue
        if not agree(rec, fresh):
            problems.append(f"{key}: the recorded comparison is not what the runs give now")
        if not fresh["ok"] and chip_role == ROLE_CHIP:
            problems.append(f"{key}: the chip run does not agree with its reference: {', '.join(fresh['failed'] or fresh['problems'])}")
    host = ref_runs.get(ROLE_HOST)
    if isinstance(host, dict):
        host_dir, ref_dir = Path(run_dir) / str(host.get("dir")), Path(run_dir) / str(ref_runs.get(ROLE_REFERENCE, {}).get("dir"))
        for key, recorded, pair in (("chip_vs_host_same_env", agreement.get("chip_vs_host_same_env"), (host_dir, Path(run_dir) / runs[ROLE_CHIP]["dir"])),
                                    ("environment_floor", res.get("environment_floor"), (ref_dir, host_dir))):
            try:
                fresh_i = compare(*pair)           # informational: recomputed and compared with the record, with no pass bar of its own
            except (ar.ReferenceError, KeyError, OSError) as exc:
                problems.append(f"{key}: {exc}")
                continue
            if not agree(recorded, fresh_i):
                problems.append(f"{key}: the recorded comparison is not what the runs give now")
    problems += repeat_problems(stage_dir, run_dir, res, ref, runs)
    problems += timing_problems(stage_dir, res, res.get("repeats"))
    if ROLE_CONTROL_CHIP in runs:
        try:
            fresh_d = control_discrimination(Path(run_dir) / ref_runs[ROLE_CONTROL_REFERENCE]["dir"], Path(run_dir) / ref_runs[ROLE_REFERENCE]["dir"],
                                             Path(run_dir) / runs[ROLE_CONTROL_CHIP]["dir"])
        except (ar.ReferenceError, KeyError) as exc:
            problems.append(f"control discrimination: {exc}")
        else:
            if not agree(res.get("control_discrimination"), fresh_d):
                problems.append("control: the recorded discrimination is not what the runs give now")
            if not fresh_d["ok"]:
                problems.append("control: the base's chip run is not nearer to the base's own reference than to the fine-tune's, so the chip path may not be using "
                                "the model's own weights")
        try:
            fresh_c = control_difference(Path(run_dir) / runs[ROLE_CHIP]["dir"], Path(run_dir) / runs[ROLE_CONTROL_CHIP]["dir"])
        except (ar.ReferenceError, KeyError) as exc:
            problems.append(f"control: {exc}")
        else:
            if not agree(res.get("control_difference"), fresh_c):
                problems.append("control: the recorded difference is not what the runs give now")
            if not fresh_c["ok"]:
                problems.append("control: the base's clip does not differ from the fine-tune's, so the chip path may not be using the model's own weights")
    return problems
