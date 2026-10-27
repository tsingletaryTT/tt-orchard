# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Records and checks the intermediate values of an audio diffusion run (design docs/superpowers/specs/2026-10-28-stable-audio-runtime-design.md,
steps 0 and 1 and section 5).

A reference run writes one directory:

    manifest.json        the model, the library, the settings, the seeds and one entry per saved array
    arrays/<name>.npy    float32 arrays, one file each (`<kind>__<name>.npy`)

Every array's sha256, shape and dtype are in the manifest, so a run can be re-checked later and two runs can be compared without trusting a file name.
The module needs numpy only. The script that makes a run from the model's own library needs torch and lives in
orchard/skills/audio-reference-templates/make_reference.py, which uses `Recorder`.

`compare_runs` checks one run (the candidate, for example the chip) against another (the reference). For each array both runs hold, it reports
three numbers: the Pearson correlation, the relative L2 error and the slope of the best-fit line of candidate against reference. PCC alone
passes an output that is scaled (the kernel probe found that on 2026-10-16), so a verdict needs all three. The bars are choices until the
noise floor of the reference itself has been measured (`bars_from_floor`); `DEFAULT_BARS` are provisional and say so.

`compare_control` is the opposite check, for the base model run beside the fine-tune with the same prompt and seed: the waveforms must
differ. Serving the wrong weights gives equal outputs, and a numerical comparison alone cannot tell that apart from success.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCHEMA = 1
MANIFEST = "manifest.json"
ARRAYS = "arrays"
KINDS = ("conditioning", "noise", "layer", "step_latent", "step_denoised", "latent", "waveform", "other")
NAME = re.compile(r"[A-Za-z0-9_.\-]{1,120}")
MAX_ARRAY_BYTES = 1 << 30            # choice: a larger array file is refused on read
MAX_MANIFEST_BYTES = 8 << 20
MAX_ARTIFACTS = 5000
# Settings that make two runs comparable. A comparison between runs that differ in any of these says nothing about a runtime.
MATCH_KEYS = ("prompt", "duration", "steps", "cfg_scale", "sampler", "seed", "dtype", "sample_size")
CONTROL_KEYS = ("prompt", "duration", "seed")      # a base control may use its own steps, sampler and scale
SETTINGS_KEYS = MATCH_KEYS + ("device",)


class ReferenceError(Exception):
    """A run directory or manifest that cannot be trusted."""


# ---- writing ----------------------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Recorder:
    """Writes arrays and the manifest of one run. Nothing is overwritten: a second array with the same kind and name is an error, and an
    existing manifest is never replaced."""

    def __init__(self, out_dir, *, model: dict, library: dict, settings: dict, seeds: dict | None = None):
        self.root = Path(out_dir)
        bad = [k for k in settings if k not in SETTINGS_KEYS]
        if bad:
            raise ReferenceError(f"unknown settings {bad}; allowed: {SETTINGS_KEYS}")
        missing = [k for k in MATCH_KEYS if k not in settings]
        if missing:
            raise ReferenceError(f"settings are missing {missing}")
        for key, value in (("model", model), ("library", library)):
            if not isinstance(value, dict) or not value:
                raise ReferenceError(f"{key} must be a non-empty object")
        if (self.root / MANIFEST).exists():
            raise ReferenceError(f"{self.root / MANIFEST} exists; a run is never overwritten")
        (self.root / ARRAYS).mkdir(parents=True, exist_ok=True)
        self.model, self.library, self.settings, self.seeds = model, library, dict(settings), dict(seeds or {})
        self.artifacts: list[dict] = []
        self._seen: set[tuple[str, str]] = set()
        self.timings: dict[str, float] = {}

    def record(self, kind: str, name: str, array, note: str | None = None) -> dict:
        """Save `array` (anything numpy can read as a float array) as float32 and note it in the manifest."""
        if kind not in KINDS:
            raise ReferenceError(f"unknown kind {kind!r}; allowed: {KINDS}")
        if not isinstance(name, str) or not NAME.fullmatch(name):
            raise ReferenceError(f"bad array name {name!r}")
        if (kind, name) in self._seen:
            raise ReferenceError(f"{kind}/{name} was already recorded")
        if len(self.artifacts) >= MAX_ARTIFACTS:
            raise ReferenceError(f"more than {MAX_ARTIFACTS} arrays")
        arr = np.asarray(array)
        if arr.dtype.kind not in "fiu" and arr.dtype.name != "bfloat16":
            raise ReferenceError(f"{kind}/{name} has dtype {arr.dtype}, which is not numeric")
        arr = np.ascontiguousarray(arr.astype(np.float32))
        rel = f"{ARRAYS}/{kind}__{name}.npy"
        fd, tmp = tempfile.mkstemp(dir=self.root / ARRAYS, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                np.save(f, arr, allow_pickle=False)
            os.replace(tmp, self.root / rel)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        entry = {"kind": kind, "name": name, "file": rel, "sha256": _sha256(self.root / rel), "shape": list(arr.shape),
                 "dtype": "float32", "finite": bool(np.isfinite(arr).all())}
        if note:
            entry["note"] = str(note)[:300]
        self.artifacts.append(entry)
        self._seen.add((kind, name))
        return entry

    def finish(self) -> Path:
        """Write the manifest, last, so a directory with a manifest holds a complete run."""
        manifest = {"schema": SCHEMA, "model": self.model, "library": self.library, "settings": self.settings,
                    "seeds": self.seeds, "timings": self.timings, "artifacts": self.artifacts}
        path = self.root / MANIFEST
        fd, tmp = tempfile.mkstemp(dir=self.root, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=1, sort_keys=True)
                f.write("\n")
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        return path


# ---- reading ----------------------------------------------------------------------------------------

@dataclass
class Run:
    root: Path
    manifest: dict
    entries: dict = field(default_factory=dict)     # (kind, name) -> entry

    @property
    def settings(self) -> dict:
        return self.manifest["settings"]

    def array(self, kind: str, name: str) -> np.ndarray:
        e = self.entries[(kind, name)]
        path = self.root / e["file"]
        if path.stat().st_size > MAX_ARRAY_BYTES:
            raise ReferenceError(f"{e['file']} is larger than {MAX_ARRAY_BYTES} bytes")
        arr = np.load(path, allow_pickle=False)
        if list(arr.shape) != e["shape"] or str(arr.dtype) != e["dtype"]:
            raise ReferenceError(f"{e['file']} has shape {list(arr.shape)} and dtype {arr.dtype}; the manifest says {e['shape']} and {e['dtype']}")
        return arr


def load_run(directory, *, verify: bool = True) -> Run:
    """Read a run directory. Refuses a manifest that is not an object of the expected shape, an array path outside `arrays/`, a link, a missing
    file and (with `verify`) a file whose sha256 differs from the manifest's."""
    root = Path(directory)
    mpath = root / MANIFEST
    try:
        if mpath.is_symlink() or mpath.stat().st_size > MAX_MANIFEST_BYTES:
            raise ReferenceError(f"{mpath} is a link or larger than {MAX_MANIFEST_BYTES} bytes")
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ReferenceError(f"cannot read {mpath}: {exc}") from exc
    except ValueError as exc:
        raise ReferenceError(f"{mpath} is not JSON: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise ReferenceError(f"{mpath} is not a schema {SCHEMA} manifest")
    for key in ("model", "library", "settings"):
        if not isinstance(manifest.get(key), dict):
            raise ReferenceError(f"manifest {key} must be an object")
    if not isinstance(manifest.get("seeds", {}), dict):
        raise ReferenceError("manifest seeds must be an object")
    missing = [k for k in MATCH_KEYS if k not in manifest["settings"]]
    if missing:
        raise ReferenceError(f"manifest settings are missing {missing}")
    items = manifest.get("artifacts")
    if not isinstance(items, list) or len(items) > MAX_ARTIFACTS:
        raise ReferenceError("manifest artifacts must be a list of at most %d entries" % MAX_ARTIFACTS)
    entries = {}
    arrays_dir = (root / ARRAYS).resolve()
    for e in items:
        if (not isinstance(e, dict) or e.get("kind") not in KINDS or not isinstance(e.get("name"), str) or not NAME.fullmatch(e["name"])
                or not isinstance(e.get("file"), str) or not re.fullmatch(r"[0-9a-f]{64}", str(e.get("sha256", "")))
                or not isinstance(e.get("shape"), list) or not all(isinstance(d, int) and not isinstance(d, bool) and d >= 0 for d in e["shape"])
                or e.get("dtype") != "float32"):
            raise ReferenceError(f"a manifest entry is malformed: {str(e)[:120]}")
        if (e["kind"], e["name"]) in entries:
            raise ReferenceError(f"{e['kind']}/{e['name']} appears twice in the manifest")
        path = root / e["file"]
        if path.is_symlink() or not path.is_file() or path.resolve().parent != arrays_dir:
            raise ReferenceError(f"{e['file']} is missing, a link, or outside {ARRAYS}/")
        if verify and _sha256(path) != e["sha256"]:
            raise ReferenceError(f"{e['file']} does not match its sha256 in the manifest")
        entries[(e["kind"], e["name"])] = e
    return Run(root, manifest, entries)


# ---- comparing --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Bars:
    pcc: float
    rel_l2: float
    slope_tol: float
    source: str = "provisional choice, not measured"


# Provisional. Layer and latent values from a bfloat16 chip against a float32 CPU are expected to differ in the third digit; the waveform is
# the least certain. `bars_from_floor` replaces these with numbers measured from the reference's own bfloat16 run.
DEFAULT_BARS = {
    "conditioning": Bars(0.99, 0.15, 0.1), "layer": Bars(0.99, 0.15, 0.1), "step_latent": Bars(0.98, 0.2, 0.15),
    "step_denoised": Bars(0.98, 0.2, 0.15), "latent": Bars(0.97, 0.25, 0.2), "waveform": Bars(0.9, 0.5, 0.3),
    "noise": Bars(1.0, 1e-6, 1e-6, "the same draws must come back unchanged"), "other": Bars(0.99, 0.15, 0.1),
}


def metrics(ref: np.ndarray, cand: np.ndarray) -> dict:
    """PCC, relative L2 error, slope of the best-fit line of `cand` against `ref`, and the largest absolute error. Non-finite values are
    reported as such and never turned into a number."""
    if ref.shape != cand.shape:
        return {"comparable": False, "reason": f"shapes differ: {list(ref.shape)} and {list(cand.shape)}"}
    x, y = ref.astype(np.float64).ravel(), cand.astype(np.float64).ravel()
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        return {"comparable": False, "reason": "a value is not finite", "ref_finite": bool(np.isfinite(x).all()), "cand_finite": bool(np.isfinite(y).all())}
    if x.size == 0:
        return {"comparable": False, "reason": "the arrays are empty"}
    xc, yc = x - x.mean(), y - y.mean()
    vx, vy = float(xc @ xc), float(yc @ yc)
    norm = float(np.linalg.norm(x))
    err = float(np.linalg.norm(y - x))
    out = {"comparable": True, "n": int(x.size), "max_abs_error": float(np.abs(y - x).max()),
           "rel_l2": (err / norm) if norm > 0 else (0.0 if err == 0 else math.inf),
           "ref_std": math.sqrt(vx / x.size), "cand_std": math.sqrt(vy / x.size)}
    if vx > 0 and vy > 0:
        out["pcc"] = float(xc @ yc) / math.sqrt(vx * vy)
        out["slope"] = float(xc @ yc) / vx
    else:
        # A constant array has no correlation. Equal constants agree; anything else does not.
        out["pcc"], out["slope"] = (1.0, 1.0) if np.array_equal(x, y) else (None, None)
    return out


def verdict(m: dict, bars: Bars) -> tuple[bool, list[str]]:
    """Whether the numbers meet `bars`, and the reasons for each one that does not."""
    if not m.get("comparable"):
        return False, [m.get("reason", "not comparable")]
    why = []
    if m["pcc"] is None:
        why.append("one array is constant and the other is not")
    elif m["pcc"] < bars.pcc:
        why.append(f"pcc {m['pcc']:.5f} is below {bars.pcc}")
    if m["rel_l2"] > bars.rel_l2:
        why.append(f"relative L2 error {m['rel_l2']:.5f} is above {bars.rel_l2}")
    if m["slope"] is None:
        pass
    elif abs(m["slope"] - 1.0) > bars.slope_tol:
        why.append(f"slope {m['slope']:.5f} is more than {bars.slope_tol} from 1 (output scale)")
    return not why, why


def settings_problems(a: Run, b: Run, keys=MATCH_KEYS) -> list[str]:
    return [f"settings.{k} differs: {a.settings.get(k)!r} and {b.settings.get(k)!r}" for k in keys if a.settings.get(k) != b.settings.get(k)]


def compare_runs(ref: Run, cand: Run, bars: dict | None = None) -> dict:
    """Compare every array both runs hold. The runs must have equal MATCH_KEYS settings. An array only one run holds is a failure: a
    candidate that skipped a step has not been checked at that step. `ok` is true only when every shared array meets its bars and none is missing."""
    bars = {**DEFAULT_BARS, **(bars or {})}
    problems = settings_problems(ref, cand)
    if problems:
        return {"ok": False, "comparable": False, "problems": problems, "results": []}
    results, missing = [], []
    for key in sorted(set(ref.entries) | set(cand.entries)):
        kind, name = key
        if key not in ref.entries or key not in cand.entries:
            missing.append(f"{kind}/{name} is only in the {'reference' if key in ref.entries else 'candidate'}")
            continue
        m = metrics(ref.array(kind, name), cand.array(kind, name))
        ok, why = verdict(m, bars[kind])
        results.append({"kind": kind, "name": name, "ok": ok, "reasons": why, "bars": bars[kind].__dict__, **m})
    if not results:
        missing.append("the runs share no array")
    return {"ok": not missing and all(r["ok"] for r in results), "comparable": True, "problems": missing, "results": results}


def log_spectrum_distance(ref: np.ndarray, cand: np.ndarray, n_fft: int = 1024, hop: int = 512) -> float | None:
    """The mean absolute difference of the log magnitude spectra, relative to the reference's mean log magnitude. A waveform's PCC falls with
    a small phase shift that nobody hears; this does not. None when either signal is shorter than one frame or not finite."""
    x, y = np.asarray(ref, dtype=np.float64), np.asarray(cand, dtype=np.float64)
    if x.shape != y.shape or not (np.isfinite(x).all() and np.isfinite(y).all()):
        return None
    x, y = x.reshape(-1, x.shape[-1]), y.reshape(-1, y.shape[-1])
    if x.shape[-1] < n_fft:
        return None
    window = np.hanning(n_fft)
    starts = range(0, x.shape[-1] - n_fft + 1, hop)

    def spec(sig):
        return np.log1p(np.abs(np.stack([np.fft.rfft(sig[:, s:s + n_fft] * window, axis=-1) for s in starts])))
    a, b = spec(x), spec(y)
    base = float(a.mean())
    return float(np.abs(a - b).mean() / base) if base > 0 else None


def bars_from_floor(floor_report: dict, margin: float = 3.0) -> dict:
    """Bars from a comparison of the reference against itself in a lower precision (`compare_runs` on a float32 run and a bfloat16 run). Each
    kind gets the worst numbers seen for it, loosened by `margin` (3 means three times the error). A kind with no result keeps the default.
    The source says where the numbers came from."""
    if not floor_report.get("comparable") or not floor_report.get("results") or margin < 1:
        raise ReferenceError("the floor report has no comparable results, or the margin is below 1")
    worst: dict[str, dict] = {}
    for r in floor_report["results"]:
        if not r.get("comparable") or r.get("pcc") is None:
            raise ReferenceError(f"{r['kind']}/{r['name']} is not comparable in the floor run, so no bar can come from it")
        w = worst.setdefault(r["kind"], {"pcc": 1.0, "rel_l2": 0.0, "slope": 0.0})
        w["pcc"], w["rel_l2"] = min(w["pcc"], r["pcc"]), max(w["rel_l2"], r["rel_l2"])
        w["slope"] = max(w["slope"], abs(r["slope"] - 1.0))
    return {kind: Bars(pcc=1.0 - margin * (1.0 - w["pcc"]), rel_l2=margin * w["rel_l2"], slope_tol=margin * w["slope"],
                       source=f"{margin} times the worst error of a float32 against a lower-precision run of the same reference")
            for kind, w in worst.items()}


def compare_control(fine_tune: Run, base: Run, min_rel_l2: float = 0.05) -> dict:
    """The waveforms of two different models run with the same prompt, duration and seed must differ by at least `min_rel_l2` (relative L2).
    Equal waveforms mean the same weights were served twice. `min_rel_l2` is a choice."""
    problems = settings_problems(fine_tune, base, CONTROL_KEYS)
    if problems:
        return {"ok": False, "comparable": False, "problems": problems}
    for run, who in ((fine_tune, "fine-tune"), (base, "base")):
        if ("waveform", "audio") not in run.entries:
            return {"ok": False, "comparable": False, "problems": [f"the {who} run has no waveform/audio array"]}
    m = metrics(fine_tune.array("waveform", "audio"), base.array("waveform", "audio"))
    if not m.get("comparable"):
        return {"ok": False, "comparable": False, "problems": [m["reason"]], **m}
    ok = m["rel_l2"] >= min_rel_l2
    return {"ok": ok, "comparable": True, "problems": [] if ok else [f"the waveforms differ by only {m['rel_l2']:.5f} (relative L2); "
                                                                      f"the bar is {min_rel_l2}: the same weights may have run twice"],
            "min_rel_l2": min_rel_l2, **m}


# ---- command line -----------------------------------------------------------------------------------

def main(argv=None, out=None) -> int:
    import argparse
    import sys
    out = out or sys.stdout
    p = argparse.ArgumentParser(prog="python -m orchard.audio_reference", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="compare a candidate run with a reference run")
    c.add_argument("reference")
    c.add_argument("candidate")
    c.add_argument("--json", action="store_true")
    k = sub.add_parser("control", help="check that a base model's run differs from the fine-tune's")
    k.add_argument("fine_tune")
    k.add_argument("base")
    k.add_argument("--min-rel-l2", type=float, default=0.05)
    v = sub.add_parser("verify", help="re-hash a run's arrays against its manifest")
    v.add_argument("run")
    args = p.parse_args(argv)
    try:
        if args.cmd == "verify":
            run = load_run(args.run)
            print(f"ok: {len(run.entries)} arrays match their sha256", file=out)
            return 0
        if args.cmd == "control":
            report = compare_control(load_run(args.fine_tune), load_run(args.base), args.min_rel_l2)
        else:
            report = compare_runs(load_run(args.reference), load_run(args.candidate))
    except ReferenceError as exc:
        print(f"refused: {exc}", file=out)
        return 2
    if getattr(args, "json", False):
        print(json.dumps(report, indent=1, default=str), file=out)
    else:
        for r in report.get("results", []):
            line = f"{'ok  ' if r['ok'] else 'FAIL'} {r['kind']}/{r['name']}"
            if r.get("comparable"):
                line += f"  pcc {r['pcc']}  rel_l2 {r['rel_l2']:.5f}  slope {r['slope']}"
            print(line + ("" if r["ok"] else "  " + "; ".join(r["reasons"])), file=out)
        for problem in report.get("problems", []):
            print("problem: " + problem, file=out)
        print("ok" if report["ok"] else "not ok", file=out)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
