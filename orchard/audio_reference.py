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
import io
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from orchard import pinned_files as pf

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


MAX_JSON_DEPTH = 32
MAX_INT_DIGITS = 30


def _strict_pairs(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate key {key!r}")
        out[key] = value
    return out


def _no_constant(name):
    raise ValueError(f"the non-finite number {name} is not allowed")


def _strict_float(text):
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"the number {text} is not finite")
    return value


def _strict_int(text):
    if len(text.lstrip("-")) > MAX_INT_DIGITS:
        raise ValueError(f"an integer of more than {MAX_INT_DIGITS} digits is not allowed")
    return int(text)


def strict_loads(text: str):
    """json.loads that refuses a repeated key (the last one would win and hide the first), NaN and infinity, integers of more than MAX_INT_DIGITS
    digits and nesting deeper than MAX_JSON_DEPTH. Raises ValueError."""
    try:
        data = json.loads(text, object_pairs_hook=_strict_pairs, parse_constant=_no_constant, parse_float=_strict_float, parse_int=_strict_int)
    except RecursionError:
        raise ValueError("nested too deeply") from None
    deepest, stack = 0, [(data, 1)]
    while stack:
        item, d = stack.pop()
        children = item.values() if isinstance(item, dict) else item if isinstance(item, list) else None
        if children is None:
            continue
        deepest = max(deepest, d)
        stack.extend((c, d + 1) for c in children)
    if deepest > MAX_JSON_DEPTH:
        raise ValueError(f"nested more than {MAX_JSON_DEPTH} levels deep")
    return data


class Recorder:
    """Writes arrays and the manifest of one run. Nothing is overwritten: a second array with the same kind and name is an error, and an
    existing manifest is never replaced.

    The run directory is created by this object and opened once; the `arrays/` directory is created and opened relative to it. Every later write goes
    through those two descriptors (orchard/pinned_files.py), so replacing the path afterwards cannot redirect a file. A file is written under a
    temporary name and published with a hard link that fails when the final name exists, so a file or a manifest that appeared in the meantime is
    never replaced."""

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
        # The directory is created here and must not exist: two commands with the same --out cannot both get it (mkdir is atomic), and the
        # files of an interrupted run are never mixed with a new one's.
        try:
            self.root.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            raise ReferenceError(f"{self.root} exists; a run is never overwritten") from None
        self._root_fd = self._arrays_fd = None
        try:
            self._root_fd = pf.open_dir(self.root, follow=False)
            os.mkdir(ARRAYS, dir_fd=self._root_fd)
            self._arrays_fd = pf.open_dir(ARRAYS, dir_fd=self._root_fd, follow=False)
        except (pf.PinnedError, OSError) as exc:
            self.close()
            raise ReferenceError(f"cannot set up {self.root}: {exc}") from exc
        self.model, self.library, self.settings, self.seeds = model, library, dict(settings), dict(seeds or {})
        self.artifacts: list[dict] = []
        self._seen: set[tuple[str, str]] = set()
        self.timings: dict[str, float] = {}
        self._finished = False

    def close(self) -> None:
        for attr in ("_arrays_fd", "_root_fd"):
            fd = getattr(self, attr, None)
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
                setattr(self, attr, None)

    def __del__(self):
        self.close()

    def record(self, kind: str, name: str, array, note: str | None = None) -> dict:
        """Save `array` (anything numpy can read as a float array) as float32 and note it in the manifest."""
        if self._finished or self._arrays_fd is None:
            raise ReferenceError("the recorder is finished; a run is never extended")
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
        final = f"{kind}__{name}.npy"
        tmp = f".{final}.tmp"
        try:
            fd = pf.create_exclusive(self._arrays_fd, tmp)
            with os.fdopen(fd, "wb") as f:
                np.save(f, arr, allow_pickle=False)
                f.flush()
                if os.fstat(f.fileno()).st_size > MAX_ARRAY_BYTES:
                    raise ReferenceError(f"{kind}/{name} would be a file larger than {MAX_ARRAY_BYTES} bytes, which the loader refuses")
                os.fsync(f.fileno())
            pf.publish_no_replace(self._arrays_fd, tmp, final)
            digest = pf.sha256_regular(self._arrays_fd, final, MAX_ARRAY_BYTES)[0]
        except pf.PinnedError as exc:
            raise ReferenceError(f"{ARRAYS}/{final}: {exc}") from exc
        finally:
            pf.unlink_quietly(self._arrays_fd, tmp)
        entry = {"kind": kind, "name": name, "file": f"{ARRAYS}/{final}", "sha256": digest, "shape": list(arr.shape),
                 "dtype": "float32", "finite": bool(np.isfinite(arr).all())}
        if note:
            entry["note"] = str(note)[:300]
        self.artifacts.append(entry)
        self._seen.add((kind, name))
        return entry

    def finish(self) -> Path:
        """Write the manifest, last, so a directory with a manifest holds a complete run. The manifest is published with a hard link that fails when
        one already exists: a second call, or a manifest someone else made, is a `ReferenceError` and leaves the existing file as it is."""
        if self._finished:
            raise ReferenceError(f"{self.root / MANIFEST} already exists; a manifest is never replaced")
        if self._root_fd is None:
            raise ReferenceError(f"the recorder for {self.root} is closed; nothing was written")
        manifest = {"schema": SCHEMA, "model": self.model, "library": self.library, "settings": self.settings,
                    "seeds": self.seeds, "timings": self.timings, "artifacts": self.artifacts}
        tmp = f".{MANIFEST}.tmp"
        try:
            fd = pf.create_exclusive(self._root_fd, tmp)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=1, sort_keys=True)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            pf.publish_no_replace(self._root_fd, tmp, MANIFEST)
            pf.fsync_dir(self._root_fd)
        except pf.PinnedError as exc:
            raise ReferenceError(f"{self.root / MANIFEST}: {exc}; a manifest is never replaced") from exc
        finally:
            pf.unlink_quietly(self._root_fd, tmp)
        self._finished = True
        self.close()
        return self.root / MANIFEST


# ---- reading ----------------------------------------------------------------------------------------

class Run:
    """A loaded run. It holds the run directory and `arrays/` open (as descriptors, `arrays/` opened without following a link), and every read is
    relative to them: replacing either path after `load_run` cannot redirect a read. `close()` (or a `with` block) releases them."""

    def __init__(self, root: Path, manifest: dict, entries: dict, root_fd: int, arrays_fd: int):
        self.root, self.manifest, self.entries = root, manifest, entries
        self._root_fd, self._arrays_fd = root_fd, arrays_fd

    @property
    def settings(self) -> dict:
        return self.manifest["settings"]

    def close(self) -> None:
        for attr in ("_arrays_fd", "_root_fd"):
            fd = getattr(self, attr, None)
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
                setattr(self, attr, None)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()

    def array(self, kind: str, name: str) -> np.ndarray:
        """The array, read once: the bytes are opened relative to the pinned `arrays/` descriptor without following a link, capped in size, hashed
        against the manifest and only then parsed, so a file that changed or was swapped for a link after `load_run` is refused and what was hashed
        is what is parsed."""
        if self._arrays_fd is None:
            raise ReferenceError("the run is closed")
        e = self.entries[(kind, name)]
        try:
            data = pf.read_regular(self._arrays_fd, Path(e["file"]).name, MAX_ARRAY_BYTES)
        except pf.PinnedError as exc:
            raise ReferenceError(f"{e['file']}: {exc}") from exc
        if hashlib.sha256(data).hexdigest() != e["sha256"]:
            raise ReferenceError(f"{e['file']} does not match its sha256 in the manifest")
        try:
            arr = np.load(io.BytesIO(data), allow_pickle=False)
        except Exception as exc:                  # a payload that is not an .npy array raises ValueError, OSError, EOFError or a zip error
            raise ReferenceError(f"{e['file']} is not a readable array: {type(exc).__name__}: {str(exc)[:120]}") from exc
        if not isinstance(arr, np.ndarray):
            raise ReferenceError(f"{e['file']} is not a readable array: it decoded to {type(arr).__name__}, not an ndarray")
        if list(arr.shape) != e["shape"] or str(arr.dtype) != e["dtype"]:
            raise ReferenceError(f"{e['file']} has shape {list(arr.shape)} and dtype {arr.dtype}; the manifest says {e['shape']} and {e['dtype']}")
        return arr


def load_run(directory, *, verify: bool = True) -> Run:
    """Read a run directory. The directory is opened once (it may be reached through a link: the operator named it) and `arrays/` is opened relative to it
    and must not be a link; the manifest and every array are read relative to those descriptors. Refuses a manifest that is not an object of the
    expected shape, a link, a missing file and (with `verify`) a file whose sha256 differs from the manifest's or that is over the size cap."""
    root = Path(directory)
    root_fd = arrays_fd = None
    try:
        try:
            root_fd = pf.open_dir(root, follow=True)
            raw = pf.read_regular(root_fd, MANIFEST, MAX_MANIFEST_BYTES)
            arrays_fd = pf.open_dir(ARRAYS, dir_fd=root_fd, follow=False)
        except pf.PinnedError as exc:
            raise ReferenceError(f"cannot read {root}: {exc}") from exc
        try:
            manifest = strict_loads(raw.decode("utf-8"))
        except ValueError as exc:                 # UnicodeDecodeError is a ValueError
            raise ReferenceError(f"{root / MANIFEST} is not JSON: {exc}") from exc
        if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
            raise ReferenceError(f"{root / MANIFEST} is not a schema {SCHEMA} manifest")
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
        for e in items:
            if (not isinstance(e, dict) or e.get("kind") not in KINDS or not isinstance(e.get("name"), str) or not NAME.fullmatch(e["name"])
                    or not isinstance(e.get("file"), str) or not re.fullmatch(r"[0-9a-f]{64}", str(e.get("sha256", "")))
                    or not isinstance(e.get("shape"), list) or not all(isinstance(d, int) and not isinstance(d, bool) and d >= 0 for d in e["shape"])
                    or e.get("dtype") != "float32"):
                raise ReferenceError(f"a manifest entry is malformed: {str(e)[:120]}")
            if (e["kind"], e["name"]) in entries:
                raise ReferenceError(f"{e['kind']}/{e['name']} appears twice in the manifest")
            if e["file"] != f"{ARRAYS}/{e['kind']}__{e['name']}.npy":
                raise ReferenceError(f"{e['file']} must be {ARRAYS}/{e['kind']}__{e['name']}.npy")
            # Existence and type are checked whether or not the digest is: only the hashing depends on `verify`.
            try:
                if verify:
                    digest = pf.sha256_regular(arrays_fd, Path(e["file"]).name, MAX_ARRAY_BYTES)[0]
                else:
                    pf.check_regular(arrays_fd, Path(e["file"]).name)
            except pf.PinnedError as exc:
                raise ReferenceError(f"{e['file']}: {exc}") from exc
            if verify and digest != e["sha256"]:
                raise ReferenceError(f"{e['file']} does not match its sha256 in the manifest")
            entries[(e["kind"], e["name"])] = e
        run = Run(root, manifest, entries, root_fd, arrays_fd)
        root_fd = arrays_fd = None                # now owned by the run
        return run
    finally:
        for fd in (arrays_fd, root_fd):
            if fd is not None:
                os.close(fd)


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


def compare_runs(ref: Run, cand: Run, bars: dict | None = None, *, cross_precision: bool = False) -> dict:
    """Compare every array both runs hold. The runs must have equal MATCH_KEYS settings. An array only one run holds is a failure: a
    candidate that skipped a step has not been checked at that step. `ok` is true only when every shared array meets its bars and none is missing.
    By default the `dtype` setting must match too. `cross_precision=True` leaves `dtype` out of that check, for the comparison the design needs
    (a float32 reference against a bfloat16 run of the same model, which is also how the noise floor is measured); every other setting must still match."""
    bars = {**DEFAULT_BARS, **(bars or {})}
    keys = tuple(k for k in MATCH_KEYS if not (cross_precision and k == "dtype"))
    problems = settings_problems(ref, cand, keys)
    if problems:
        return {"ok": False, "comparable": False, "cross_precision": cross_precision, "problems": problems, "results": []}
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
    return {"ok": not missing and all(r["ok"] for r in results), "comparable": True, "cross_precision": cross_precision, "problems": missing,
            "results": results}


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


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def bars_from_floor(floor_report: dict, margin: float = 3.0) -> dict:
    """Bars from a comparison of the reference against itself in a lower precision (`compare_runs(..., cross_precision=True)` on a float32 run
    and a bfloat16 run). Each kind gets the worst numbers seen for it, loosened by `margin` (3 means three times the error). A kind with no
    result keeps the default. The source says where the numbers came from.

    The report is refused when it is malformed, was not a cross-precision comparison, lists a missing or extra array, holds a result that could
    not be compared, or holds a `noise` result that failed: the two runs then did not use the same noise draws, so their differences say nothing
    about precision, and every other number in the report is suspect. Noise bars are never derived from a floor; they stay at their defaults."""
    if not isinstance(floor_report, dict) or not _number(margin) or margin < 1:
        raise ReferenceError("the floor report must be a comparison report and the margin a number of at least 1")
    if floor_report.get("cross_precision") is not True:
        raise ReferenceError("the floor report is not a cross-precision comparison (compare_runs(..., cross_precision=True))")
    if not floor_report.get("comparable") or floor_report.get("problems"):
        raise ReferenceError(f"the floor report is not comparable or lists problems: {str(floor_report.get('problems'))[:200]}")
    results = floor_report.get("results")
    if not isinstance(results, list) or not results:
        raise ReferenceError("the floor report has no results")
    worst: dict[str, dict] = {}
    for r in results:
        if not isinstance(r, dict) or r.get("kind") not in KINDS or not isinstance(r.get("name"), str):
            raise ReferenceError(f"a floor result is malformed: {str(r)[:120]}")
        label = f"{r['kind']}/{r['name']}"
        if r.get("comparable") is not True or not all(_number(r.get(k)) for k in ("pcc", "rel_l2", "slope")):
            raise ReferenceError(f"{label} is not comparable in the floor run, so no bar can come from it")
        if r["kind"] == "noise":
            if r.get("ok") is not True:
                raise ReferenceError(f"{label} failed its noise check: the two floor runs did not use the same noise draws")
            continue
        w = worst.setdefault(r["kind"], {"pcc": 1.0, "rel_l2": 0.0, "slope": 0.0})
        w["pcc"], w["rel_l2"] = min(w["pcc"], r["pcc"]), max(w["rel_l2"], r["rel_l2"])
        w["slope"] = max(w["slope"], abs(r["slope"] - 1.0))
    return {kind: Bars(pcc=1.0 - margin * (1.0 - w["pcc"]), rel_l2=margin * w["rel_l2"], slope_tol=margin * w["slope"],
                       source=f"{margin} times the worst error of a float32 against a lower-precision run of the same reference")
            for kind, w in worst.items()}


def compare_control(fine_tune: Run, base: Run, min_rel_l2: float = 0.05, *, require_matched: bool = True) -> dict:
    """The waveforms of two different models run with the same prompt, duration and seed must differ by at least `min_rel_l2` (relative L2).
    Equal waveforms mean the same weights were served twice. `min_rel_l2` is a choice.

    By default every MATCH_KEYS setting must be equal too (steps, sampler, guidance scale, dtype and sample size, as well as prompt, duration and
    seed): the base run is the base weights with the fine-tune's settings. With different settings a wrong checkpoint cannot be told from a right
    one, because a fine-tune accidentally loaded with the base weights still differs from a base run made with the base's own recommended settings.
    `require_matched=False` allows that second kind of comparison (prompt, duration and seed must still be equal) and says in the report that it cannot
    show a wrong checkpoint."""
    problems = settings_problems(fine_tune, base, CONTROL_KEYS)
    matched_problems = settings_problems(fine_tune, base, MATCH_KEYS)
    matched = not matched_problems
    if problems:
        return {"ok": False, "comparable": False, "matched": matched, "problems": problems}
    if require_matched and not matched:
        return {"ok": False, "comparable": False, "matched": False,
                "problems": matched_problems + ["a control needs matched settings: run the base weights with the fine-tune's settings (or ask for an "
                                                "unmatched comparison, which cannot show a wrong checkpoint)"]}
    for run, who in ((fine_tune, "fine-tune"), (base, "base")):
        if ("waveform", "audio") not in run.entries:
            return {"ok": False, "comparable": False, "matched": matched, "problems": [f"the {who} run has no waveform/audio array"]}
    m = metrics(fine_tune.array("waveform", "audio"), base.array("waveform", "audio"))
    if not m.get("comparable"):
        return {"ok": False, "comparable": False, "matched": matched, "problems": [m["reason"]], **m}
    ok = m["rel_l2"] >= min_rel_l2
    out = {"ok": ok, "comparable": True, "matched": matched,
           "problems": [] if ok else [f"the waveforms differ by only {m['rel_l2']:.5f} (relative L2); the bar is {min_rel_l2}: the same weights may have run twice"],
           "min_rel_l2": min_rel_l2, **m}
    if not matched:
        out["note"] = ("the settings differ between the two runs (" + "; ".join(matched_problems)
                       + "), so this comparison cannot show that the right checkpoint was served")
    return out


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
    c.add_argument("--cross-precision", action="store_true", help="let the dtype setting differ (float32 reference, bfloat16 candidate)")
    k = sub.add_parser("control", help="check that a base model's run differs from the fine-tune's")
    k.add_argument("fine_tune")
    k.add_argument("base")
    k.add_argument("--min-rel-l2", type=float, default=0.05)
    k.add_argument("--unmatched", action="store_true", help="allow different steps, sampler or guidance scale (cannot show a wrong checkpoint)")
    v = sub.add_parser("verify", help="re-hash a run's arrays against its manifest")
    v.add_argument("run")
    args = p.parse_args(argv)
    try:
        if args.cmd == "verify":
            with load_run(args.run) as run:
                print(f"ok: {len(run.entries)} arrays match their sha256", file=out)
            return 0
        if args.cmd == "control":
            with load_run(args.fine_tune) as ft, load_run(args.base) as base:
                report = compare_control(ft, base, args.min_rel_l2, require_matched=not args.unmatched)
        else:
            with load_run(args.reference) as ref, load_run(args.candidate) as cand:
                report = compare_runs(ref, cand, cross_precision=args.cross_precision)
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
        if report.get("note"):
            print("note: " + report["note"], file=out)
        print("ok" if report["ok"] else "not ok", file=out)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
