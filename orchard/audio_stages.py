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

Bars. The agreement bars are the provisional ones in `audio_reference.DEFAULT_BARS`. The result says so. The sound checks (duration, level,
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

from orchard import audio_reference as ar
from orchard import safeio
from orchard.audio_runtime import REFERENCE_FILE, RESULT_FILE, STAGE_DISK_GB

SCHEMA = 1
PROMPT = "A warm acoustic guitar melody with soft percussion, 100 BPM"
DURATION_S = 10.0
SEED = 42
SAMPLE_SIZE = 5292032                  # the library's own default for `generate`
SAMPLE_RATE = 44100

# Library registry names and the settings each is run with. The fine-tune is the post-trained 8-step model (ping-pong sampler, guidance scale 1);
# its base is the control and needs the base's own settings (the audio design's section 5).
LIBRARY_NAMES = {"stabilityai/stable-audio-3-small-music": "small-music", "stabilityai/stable-audio-3-small-music-base": "small-music-base"}
MODEL_SETTINGS = {"small-music": {"steps": 8, "cfg_scale": 1.0, "sampler": "pingpong"},
                  "small-music-base": {"steps": 50, "cfg_scale": 7.0, "sampler": "euler"}}

# Sound checks. Source: the audio design, section 6 (24 clips measured on 2026-10-15).
DURATION_TOLERANCE = 0.05
LOUD_WINDOW_MS = 100
LOUD_DBFS = -40.0
MIN_LOUD_FRACTION = 0.40
MAX_FULL_SCALE_FRACTION = 0.01
FULL_SCALE = 0.9999

ROLE_REFERENCE, ROLE_CONTROL_REFERENCE, ROLE_HOST = "reference", "control_reference", "host_same_env"
ROLE_CHIP, ROLE_CONTROL_CHIP = "chip", "control_chip"


class AudioStageError(Exception):
    """A problem a stage reports as its failure reason."""


def library_name(repo) -> str | None:
    """The library's registry name for a repository id (a revision after `@` is ignored), or None for a model this module does not know."""
    return LIBRARY_NAMES.get(str(repo or "").split("@", 1)[0])


def settings_for(name: str) -> dict:
    """The recorded settings (audio_reference.MATCH_KEYS) of the run of library model `name`."""
    s = MODEL_SETTINGS[name]
    return {"prompt": PROMPT, "duration": DURATION_S, "steps": s["steps"], "cfg_scale": s["cfg_scale"], "sampler": s["sampler"], "seed": SEED,
            "sample_size": SAMPLE_SIZE}


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


def write_clip(run_dir, sample_rate: int = SAMPLE_RATE) -> Path:
    """`clips/audio.flac` of a recorded run, written from its waveform array (a run that already has one keeps it). The array's bytes are
    re-hashed by `load_run`, so the clip is the recorded waveform."""
    import soundfile
    clip = Path(run_dir) / "clips" / "audio.flac"
    if clip.is_file():
        return clip
    with ar.load_run(run_dir) as run:
        wave = np.asarray(run.array("waveform", "audio"))
    clip.parent.mkdir(exist_ok=True)
    soundfile.write(str(clip), wave.T, sample_rate)
    return clip


def clip_matches_waveform(run_dir, wave, sample_rate: int = SAMPLE_RATE) -> list[str]:
    """Reasons the clip file is not the recorded waveform: unreadable, or another length, rate or channel count. The FLAC is lossless on 16-bit
    samples, so the decoded values are compared with the waveform to within one 16-bit step."""
    clip = Path(run_dir) / "clips" / "audio.flac"
    try:
        import soundfile
        data, rate = soundfile.read(str(clip), always_2d=True, dtype="float64")
    except Exception as exc:                                    # a missing file, a bad file or no soundfile
        return [f"the clip {clip.name} cannot be read: {type(exc).__name__}"]
    w = np.asarray(wave, dtype=np.float64)
    if rate != sample_rate or data.shape != w.T.shape:
        return [f"the clip is {data.shape[1]} channels, {data.shape[0]} samples at {rate} Hz; the waveform is {w.shape[0]} channels, {w.shape[1]} samples"]
    if float(np.abs(data - w.T).max()) > 2.0 / 32768:
        return ["the clip's samples differ from the recorded waveform by more than one 16-bit step"]
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


def compare(ref_dir, cand_dir, *, informational: bool = False) -> dict:
    """`summarise(compare_runs(...))` of two run directories, float32 reference against the candidate with the dtype setting left out of the
    match, and the waveform's log-spectrum distance added. `informational` says the numbers are reported and no bar decides anything (the base
    control: fifty guided steps compound the same per-pass error the fine-tune's eight steps show)."""
    with ar.load_run(ref_dir) as ref, ar.load_run(cand_dir) as cand:
        out = summarise(ar.compare_runs(ref, cand, cross_precision=True))
        if informational:
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
        return {"dir": rel, "model": m["model"]["name"], "revision": m["model"]["revision"], "library_commit": m["library"]["commit"],
                "settings": dict(m["settings"]), "timings": dict(m.get("timings", {})),
                "manifest_sha256": sha256_file(d / "manifest.json")}


# ---- the files a stage writes and its gate re-derives ----------------------------------------------------------------------------

def load_json(path, what: str) -> dict:
    try:
        return safeio.read_json(path, max_bytes=4 << 20, expect=dict)
    except (OSError, ValueError, safeio.SafeReadError) as exc:
        raise AudioStageError(f"{what} cannot be read: {exc}") from exc


def inside(run_dir, rel) -> Path | None:
    """`rel` as a path inside `run_dir`, or None for anything that leaves it: an absolute path, a `..` part, or a link on the way."""
    if not isinstance(rel, str) or not rel or Path(rel).is_absolute() or ".." in Path(rel).parts:
        return None
    p = Path(run_dir)
    for part in Path(rel).parts:
        p = p / part
        if p.is_symlink():
            return None
    return p


def check_run_dir(run_dir, entry: dict, want_settings: dict | None, label: str, problems: list) -> dict | None:
    """Verify one recorded run named by a result entry: it is inside the run directory, its arrays match their hashes, its manifest hash is the
    recorded one, its settings are the expected ones and its waveform passes the sound checks (and the clip is that waveform). Returns the
    sound-check record, or None after adding to `problems`."""
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
    if want_settings is not None:
        got = {k: facts["settings"].get(k) for k in want_settings}
        if got != want_settings:
            problems.append(f"{label}: its settings are {got}, not the expected {want_settings}")
    checks = sound_checks(wave, SAMPLE_RATE, float(facts["settings"].get("duration", 0)))
    if not checks["ok"]:
        problems.append(f"{label}: the clip fails the sound checks: {'; '.join(checks['reasons'])}")
    problems += [f"{label}: {x}" for x in clip_matches_waveform(p, wave)]
    return checks


def agree(recorded, fresh) -> bool:
    """A recorded comparison summary equals one computed again now (numbers to 1e-9, everything else exactly)."""
    if isinstance(recorded, dict) and isinstance(fresh, dict):
        return recorded.keys() == fresh.keys() and all(agree(recorded[k], fresh[k]) for k in recorded)
    if isinstance(recorded, (int, float)) and isinstance(fresh, (int, float)) and not isinstance(recorded, bool) and not isinstance(fresh, bool):
        return math.isclose(recorded, fresh, rel_tol=1e-9, abs_tol=1e-9)
    if isinstance(recorded, list) and isinstance(fresh, list):
        return len(recorded) == len(fresh) and all(agree(a, b) for a, b in zip(recorded, fresh))
    return type(recorded) is type(fresh) and recorded == fresh


# ---- stage 1: the CPU references -------------------------------------------------------------------------------------------------



def reference_plan(model_id, base_id) -> list[dict]:
    """The runs stage 1 makes: [{role, name (library name), env ('reference' or 'chip' interpreter), dir}]. The control and the same-environment
    run exist only when the base is a model this module knows."""
    name = library_name(model_id)
    if name is None:
        raise AudioStageError(f"{model_id} is not a model the audio runtime knows ({sorted(LIBRARY_NAMES)})")
    plan = [{"role": ROLE_REFERENCE, "name": name, "interpreter": "reference", "dir": f"reference/{name}"},
            {"role": ROLE_HOST, "name": name, "interpreter": "chip", "dir": f"host/{name}"}]
    base = library_name(base_id)
    if base is not None and base != name:
        plan.append({"role": ROLE_CONTROL_REFERENCE, "name": base, "interpreter": "reference", "dir": f"reference/{base}"})
    return plan


def reference_command(item: dict, paths, stage_rel_dir: Path, script: Path) -> tuple[list[str], dict]:
    """argv and extra environment for one CPU run. The reference interpreter finds the library through its editable install; the chip interpreter
    through PYTHONPATH. `--expect-src` makes the script refuse a library from anywhere else."""
    s = settings_for(item["name"])
    python = paths.reference_python if item["interpreter"] == "reference" else paths.chip_python
    argv = [python, str(script), "--model", item["name"], "--out", str(stage_rel_dir / item["dir"]), "--prompt", s["prompt"],
            "--duration", str(s["duration"]), "--steps", str(s["steps"]), "--cfg-scale", str(s["cfg_scale"]), "--seed", str(s["seed"]),
            "--sampler", s["sampler"], "--sample-size", str(s["sample_size"]), "--expect-src", paths.src_dir]
    env = {"PYTHONPATH": paths.src_dir} if item["interpreter"] == "chip" else {}
    return argv, env


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
        sound[role] = check_run_dir(run_dir, entry, settings_for(name), role, problems)
    if ROLE_REFERENCE in runs and ROLE_HOST in runs and not problems:
        a, b = inside(run_dir, runs[ROLE_REFERENCE].get("dir")), inside(run_dir, runs[ROLE_HOST].get("dir"))
        try:
            fresh = compare(a, b)
        except ar.ReferenceError as exc:
            return problems + [f"environment floor: {exc}"]
        if not agree(rec.get("environment_floor"), fresh):
            problems.append("the recorded environment floor is not what the two runs give now")
    return problems


def cached_run(cache_dir, name: str, revision: str | None, library_commit: str | None):
    """A saved CPU reference of library model `name` that can stand in for a fresh one, or None: the directory loads and its arrays match their
    hashes, its model revision and library commit are the wanted ones and its settings are the stage's settings for that model. A reference is a
    deterministic float32 CPU run, so the same inputs give the same arrays (measured: a repeat run is bit-identical)."""
    if not cache_dir or not revision or not library_commit:
        return None
    d = Path(cache_dir) / name
    if not d.is_dir() or d.is_symlink():
        return None
    try:
        with ar.load_run(d) as run:
            m = run.manifest
            ok = (m["model"]["name"] == name and m["model"]["revision"] == revision and m["library"]["commit"] == library_commit
                  and {k: m["settings"].get(k) for k in settings_for(name)} == settings_for(name) and m["settings"].get("dtype") == "float32"
                  and m["settings"].get("device") == "cpu")
    except (ar.ReferenceError, OSError, KeyError, TypeError):
        return None
    return d if ok else None


def run_references(stage_dir, run_dir, paths, plan, runner, *, revisions: dict, library_commit: str | None, cache_dir=None,
                   script: Path | None = None, timeout_s: float = 3600.0, say=lambda text: None) -> dict:
    """Make every run of `plan` under `stage_dir` and write `audio_reference.json`. `runner(argv, env_extra, timeout_s, log_path)` runs one command
    and returns its exit status (None for a timeout); `revisions` maps a library name to the snapshot revision wanted. Raises AudioStageError when
    a run fails. A reference (not the same-environment run) that the cache holds is copied instead of made again."""
    from orchard import audio_runtime
    script = script or audio_runtime.REFERENCE_SCRIPT
    stage_dir, run_dir = Path(stage_dir), Path(run_dir)
    rel_stage = stage_dir.resolve().relative_to(run_dir.resolve())
    (stage_dir / "log").mkdir(parents=True, exist_ok=True)
    runs = {}
    for item in plan:
        out = stage_dir / item["dir"]
        out.parent.mkdir(parents=True, exist_ok=True)
        hit = cached_run(cache_dir, item["name"], revisions.get(item["name"]), library_commit) if item["interpreter"] == "reference" else None
        t0 = time.time()
        if hit is not None:
            shutil.copytree(hit, out, symlinks=False)
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
            write_clip(out)
            facts = run_facts(run_dir, str(rel_stage / item["dir"]))
        except (ar.ReferenceError, OSError, KeyError) as exc:
            raise AudioStageError(f"the CPU run for {item['role']} cannot be read back: {exc}") from exc
        runs[item["role"]] = {**facts, "role": item["role"], "interpreter": item["interpreter"], "from_cache": from_cache, "wall_seconds": seconds,
                              "clip": f"{facts['dir']}/clips/audio.flac"}
    rec = {"schema": SCHEMA, "runs": runs,
           "environment_floor": compare(Path(run_dir) / runs[ROLE_REFERENCE]["dir"], Path(run_dir) / runs[ROLE_HOST]["dir"]),
           "limits": ["The reference is the library's own float32 CPU run. The same-environment run is the same code in the interpreter that the chip "
                      "run uses; the difference between the two is what a change of CPU environment does to these numbers, before any chip is involved."]}
    write_json(stage_dir / REFERENCE_FILE, rec)
    return rec


def write_json(path, value) -> None:
    """Write a JSON file whole or not at all."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


# ---- stage 2: the clip on a chip ------------------------------------------------------------------------------------------------

CONFIG_FILE = "audio_chip_config.json"


def chip_plan(model_id, base_id) -> list[dict]:
    name = library_name(model_id)
    if name is None:
        raise AudioStageError(f"{model_id} is not a model the audio runtime knows")
    plan = [{"role": ROLE_CHIP, "name": name}]
    base = library_name(base_id)
    if base is not None and base != name:
        plan.append({"role": ROLE_CONTROL_CHIP, "name": base})
    return plan


def write_chip_config(stage_dir, paths, hf_home: str, plan: list[dict]) -> Path:
    """`audio_chip_config.json`, the input of audio_chip_test.py: shared settings and one entry per run."""
    runs = [{"name": item["name"], **{k: MODEL_SETTINGS[item["name"]][k] for k in ("steps", "cfg_scale", "sampler")}} for item in plan]
    for r, item in zip(runs, plan):
        r["model"] = item["name"]
    cfg = {"prompt": PROMPT, "duration": DURATION_S, "seed": SEED, "sample_size": SAMPLE_SIZE, "src_dir": paths.src_dir, "hf_home": hf_home,
           "out_dir": "chip", "runs": runs}
    path = Path(stage_dir) / CONFIG_FILE
    write_json(path, cfg)
    return path


def chip_test_command(paths, stage_dir_abs: Path, script: Path) -> str:
    """The one command the hardware phase runs under the lease."""
    return f"{paths.chip_python} {script} --config {stage_dir_abs / CONFIG_FILE}"


def finish_chip(stage_dir, run_dir, ref_stage_dir, plan: list[dict], reference_roles: dict) -> dict:
    """After the hardware test: compare each chip run with its reference (and the fine-tune's with the same-environment run), check the control
    and the clips, and write `audio_result.json`. `reference_roles` maps a chip role to the reference role it is compared with. Raises
    AudioStageError when a recorded run is missing."""
    stage_dir, run_dir, ref_stage_dir = Path(stage_dir), Path(run_dir), Path(ref_stage_dir)
    ref_rec = load_json(ref_stage_dir / REFERENCE_FILE, REFERENCE_FILE)
    rel = stage_dir.resolve().relative_to(run_dir.resolve())
    runs, agreement, sound = {}, {}, {}
    for item in plan:
        d = stage_dir / "chip" / item["name"]
        if not (d / "manifest.json").is_file():
            raise AudioStageError(f"the chip run for {item['role']} ({item['name']}) left no manifest")
        facts = run_facts(run_dir, str(rel / "chip" / item["name"]))
        try:
            clip = write_clip(d)
        except (ar.ReferenceError, OSError) as exc:
            raise AudioStageError(f"the chip clip for {item['name']} cannot be written: {exc}") from exc
        chip_json = {}
        try:
            chip_json = load_json(d / "chip.json", "chip.json")
        except AudioStageError:
            pass
        runs[item["role"]] = {**facts, "role": item["role"], "clip": f"{facts['dir']}/clips/audio.flac", "device": chip_json.get("device"),
                              "total_seconds": chip_json.get("total_s")}
        ref_role = reference_roles[item["role"]]
        ref_entry = ref_rec["runs"][ref_role]
        agreement[f"{item['role']}_vs_{ref_role}"] = compare(run_dir / ref_entry["dir"], d, informational=item["role"] == ROLE_CONTROL_CHIP)
    if ROLE_CHIP in runs and ROLE_HOST in ref_rec["runs"]:
        agreement["chip_vs_host_same_env"] = compare(run_dir / ref_rec["runs"][ROLE_HOST]["dir"], stage_dir / "chip" / plan[0]["name"])
    control = discrimination = None
    if ROLE_CONTROL_CHIP in runs:
        control = control_difference(stage_dir / "chip" / plan[0]["name"], stage_dir / "chip" / plan[1]["name"])
        discrimination = control_discrimination(run_dir / ref_rec["runs"][ROLE_CONTROL_REFERENCE]["dir"], run_dir / ref_rec["runs"][ROLE_REFERENCE]["dir"],
                                                stage_dir / "chip" / plan[1]["name"])
    result = {"schema": SCHEMA, "runtime": "stable-audio-3-dit-on-chip", "settings": settings_for(plan[0]["name"]), "runs": runs,
              "reference_runs": ref_rec["runs"], "agreement": agreement, "environment_floor": ref_rec.get("environment_floor"),
              "control_difference": control, "control_discrimination": discrimination,
              "limits": ["Nobody has listened to the clips; no check here says the music is good.",
                         "The agreement bars are provisional choices, not yet set from a measured noise floor.",
                         "Only the diffusion transformer ran on the chip. The T5Gemma encoder, the sampler and the autoencoder decoder ran on the host.",
                         "Agreement is measured on one prompt, one duration and one seed."]}
    write_json(stage_dir / RESULT_FILE, result)
    return result


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
    if not isinstance(runs, dict) or not isinstance(ref_runs, dict) or not isinstance(agreement, dict):
        return [f"{RESULT_FILE} needs runs, reference_runs and agreement"]
    if ROLE_CHIP not in runs:
        return [f"{RESULT_FILE} has no chip run"]
    if not agree(res.get("reference_runs"), ref_runs):
        problems.append("the result's reference_runs are not stage 1's")
    for role, entry in runs.items():
        name = entry.get("model") if isinstance(entry, dict) else None
        if name not in MODEL_SETTINGS:
            problems.append(f"{role}: model {name!r} is not one the audio runtime knows")
            continue
        check_run_dir(run_dir, entry, {**settings_for(name)}, role, problems)
    pairs = [(ROLE_CHIP, ROLE_REFERENCE)] + ([(ROLE_CONTROL_CHIP, ROLE_CONTROL_REFERENCE)] if ROLE_CONTROL_CHIP in runs else [])
    for chip_role, ref_role in pairs:
        key = f"{chip_role}_vs_{ref_role}"
        if ref_role not in ref_runs:
            problems.append(f"stage 1 made no {ref_role} run to compare {chip_role} with")
            continue
        rec = agreement.get(key)
        try:
            fresh = compare(Path(run_dir) / ref_runs[ref_role]["dir"], Path(run_dir) / runs[chip_role]["dir"], informational=chip_role == ROLE_CONTROL_CHIP)
        except (ar.ReferenceError, KeyError) as exc:
            problems.append(f"{key}: {exc}")
            continue
        if not agree(rec, fresh):
            problems.append(f"{key}: the recorded comparison is not what the runs give now")
        if not fresh["ok"] and chip_role == ROLE_CHIP:
            problems.append(f"{key}: the chip run does not agree with its reference: {', '.join(fresh['failed'] or fresh['problems'])}")
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
