# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Agreement bars set from a measured noise floor, with a known-bad control (issue 173).

`audio_reference.DEFAULT_BARS` are choices made before anything was measured. This module measures the spread between recorded CPU runs of the
same code and the same seeds, sets each bar to a stated multiple of that spread, and then asks whether the bars can tell a good run from a damaged
one. A bar that cannot is reported as unusable. It works on recorded runs only (the layout of `orchard/audio_reference.py`, read through
`audio_reference.load_run`); it needs no chip and no model.

Steps, in `analyse`:
  1. `measure_floor`: for every array that all the floor runs hold, the agreement of every pair of runs. The runs are CPU runs in the reference
     environment (two or more are better: a repeat shows whether a single environment is deterministic) and CPU runs of the same code in the chip's
     own Python environment. Pairs inside one environment and pairs across the two are kept apart; the floor of an array is the worst pair.
  2. `derive_bars`: per group of arrays (by kind, or by kind and name family), the bar for relative L2 error is `margin` times the worst floor in the
     group, the bar for PCC is `1 - margin^2 * (1 - worst PCC)` (the shortfall of PCC grows with the square of the error), and the bar for the slope is `margin` times the worst distance of a slope from 1. A floor
     of zero is lifted to `MIN_FLOOR`. Noise arrays are never given a floor: the same draws must come back unchanged.
  3. `evaluate` the good run (a real candidate run, for example the chip's) against the reference with those bars.
  4. The known-bad controls: the good run with some arrays damaged (`make_damaged_run`: rounded to bfloat16, or given an error of a stated relative
     size), and any real bad run the caller has. Each must fail the bars. The good run must pass.
  5. `detectable`: for each group, how much extra relative error the bars would let through unseen.
The bars are usable when the good run passes and every control fails. The result holds the floor, the margin, the bars, the controls and the reasons
for any refusal; `bars_or_fallback` hands back the measured bars, or the provisional ones only when the caller asks for the fallback, with a label
that says why.

What a CPU-only floor does not contain: the chip's own arithmetic. A bfloat16 chip run is expected to sit above the CPU environment floor on
many arrays, so the margin has to leave room for it. The good-run check measures exactly that, and the `detectable` table says how much room is left
for damage to hide in.
"""
from __future__ import annotations

import hashlib
import math
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from orchard import audio_reference as ar
from orchard import first_divergence as fd

ReferenceError = ar.ReferenceError


class NoiseFloorError(Exception):
    """The runs cannot give a floor, an option is unusable, or a result is not trustworthy."""


DEFAULT_MARGIN = 3.0       # choice: matches audio_reference.bars_from_floor
MIN_FLOOR = 1e-6           # choice: a pair of identical runs has a floor of zero, which would give a bar nobody can meet
GROUPS = ("kind", "family")
SLOPE_RULES = ("floor", "rel_l2")
NOISE_PCC_BAR = 1.0 - 1e-9


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def group_of(kind: str, name: str, by: str) -> str:
    """The group of an array: its kind, or its kind and its name with each run of digits replaced by `#` (layer/layer_#, step_latent/#)."""
    if by == "kind":
        return kind
    return f"{kind}/{re.sub(r'[0-9]+', '#', name)}"


# ---- measuring the floor ----------------------------------------------------------------------------------------

def _describe(run, index: int) -> dict:
    root = getattr(run, "root", None)
    h = hashlib.sha256()
    for (kind, name), e in sorted(run.entries.items()):
        h.update(f"{kind}/{name}:{e.get('sha256', '')}\n".encode())
    return {"name": Path(root).name if root is not None else f"run{index}", "arrays_sha256": h.hexdigest()}


def _pair(a: np.ndarray, b: np.ndarray, label: str) -> dict:
    m = ar.metrics(a, b)
    if not m.get("comparable"):
        raise NoiseFloorError(f"{label} cannot be compared between two floor runs: {m.get('reason')}")
    if m["pcc"] is None:
        raise NoiseFloorError(f"{label} is constant in one floor run and not in the other")
    return {"pcc": m["pcc"], "rel_l2": m["rel_l2"], "slope_dev": abs(m["slope"] - 1.0)}


def _worst(pairs: list[dict]) -> dict | None:
    if not pairs:
        return None
    return {"pcc": min(p["pcc"] for p in pairs), "rel_l2": max(p["rel_l2"] for p in pairs), "slope_dev": max(p["slope_dev"] for p in pairs)}


def measure_floor(reference_env, chip_env, *, group: str = "kind") -> dict:
    """The spread of every array across CPU runs. `reference_env` are runs in the reference environment (the first one is the reference itself),
    `chip_env` are runs of the same code in the chip's environment. All runs must carry equal settings, dtype included. Raises `NoiseFloorError`
    for fewer than two runs, runs that do not belong together, and noise arrays that differ."""
    if group not in GROUPS:
        raise NoiseFloorError(f"group must be one of {GROUPS}")
    reference_env, chip_env = list(reference_env), list(chip_env)
    runs = reference_env + chip_env
    env_of = [0] * len(reference_env) + [1] * len(chip_env)
    if len(runs) < 2:
        raise NoiseFloorError("a floor needs at least two runs")
    base = runs[0]
    for r in runs[1:]:
        diff = ar.settings_problems(base, r)
        if diff:
            raise NoiseFloorError("the floor runs are not the same setup: " + "; ".join(diff))
    shared = set(base.entries)
    for r in runs[1:]:
        shared &= set(r.entries)
    problems = [f"{k}/{n} is not in every floor run, so it has no floor" for (k, n) in sorted(set().union(*(set(r.entries) for r in runs)) - shared)]
    arrays, by_group = {}, {}
    for kind, name in sorted(shared):
        label = f"{kind}/{name}"
        loaded = [r.array(kind, name) for r in runs]
        within, between, every = [], [], []
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                p = _pair(loaded[i], loaded[j], label)
                every.append(p)
                (within if env_of[i] == env_of[j] else between).append(p)
        floor = _worst(every)
        if kind == "noise":
            if floor["rel_l2"] > 1e-6:
                raise NoiseFloorError(f"the noise array {label} differs between floor runs (relative L2 {floor['rel_l2']:.3g}): "
                                      "they did not use the same noise draws, so their differences say nothing about the environment")
            continue
        arrays[label] = {"kind": kind, "name": name, "within_environment": _worst(within), "between_environments": _worst(between), "floor": floor}
        by_group.setdefault(group_of(kind, name, group), []).append((name, floor))
    if not arrays:
        raise NoiseFloorError("the floor runs share no array to measure")

    def summarise_groups(items_by_group):
        out = {}
        for g, items in sorted(items_by_group.items()):
            w = _worst([f for _, f in items])
            out[g] = {"worst": w, "worst_array": max(items, key=lambda it: it[1]["rel_l2"])[0], "arrays": len(items)}
        return out
    kinds = {}
    for a in arrays.values():
        kinds.setdefault(a["kind"], []).append((a["name"], a["floor"]))
    return {"group": group, "runs": {"reference_environment": [_describe(r, i) for i, r in enumerate(reference_env)],
                                     "chip_environment": [_describe(r, len(reference_env) + i) for i, r in enumerate(chip_env)]},
            "arrays": arrays, "groups": summarise_groups(by_group), "kinds": summarise_groups(kinds), "problems": problems}


# ---- bars from the floor ----------------------------------------------------------------------------------------

def derive_bars(floor: dict, *, margin: float = DEFAULT_MARGIN, slope_rule: str = "floor") -> dict:
    """Bars per group from `measure_floor`'s result and a margin (3 means three times the worst spread). Returns {group: audio_reference.Bars}, with
    `source` naming the margin and the count of arrays. Noise arrays get the bar that their draws must come back unchanged.

    `slope_rule` says where the slope bar comes from. `floor` (the default) is `margin` times the worst distance of a slope from 1 between the CPU
    runs. A chip run has a scale error of its own (on the Stable Audio layers every output is about 1 percent small), which no CPU floor contains, so
    that rule can fail a good chip run on slope alone. `rel_l2` sets the slope bar equal to the relative L2 bar: a pure scale error of size s moves the
    relative L2 error by about s, so a scale error is still limited to the error budget. The source text of each bar names the rule."""
    if not _num(margin) or margin < 1:
        raise NoiseFloorError("the margin must be a finite number of at least 1")
    if slope_rule not in SLOPE_RULES:
        raise NoiseFloorError(f"slope_rule must be one of {SLOPE_RULES}")
    if not isinstance(floor, dict) or not isinstance(floor.get("groups"), dict) or not floor["groups"]:
        raise NoiseFloorError("the floor result has no groups")
    bars = {}
    for g, info in floor["groups"].items():
        w = info["worst"]
        # 1 - PCC grows with the square of the relative error (about r^2 / 2), so the same margin on the error is the margin squared on 1 - PCC.
        rel = margin * max(w["rel_l2"], MIN_FLOOR)
        slope = margin * max(w["slope_dev"], MIN_FLOOR) if slope_rule == "floor" else rel
        bars[g] = ar.Bars(pcc=max(-1.0, 1.0 - margin ** 2 * max(1.0 - w["pcc"], MIN_FLOOR ** 2)), rel_l2=rel, slope_tol=slope,
                          source=f"measured: {margin} times the worst spread between CPU environments over {info['arrays']} arrays; "
                                 + ("slope bar from the slope floor" if slope_rule == "floor" else "slope bar equal to the relative L2 bar"))
    bars["noise"] = ar.Bars(NOISE_PCC_BAR, 1e-6, 1e-6, "the same draws must come back unchanged")
    return bars


def evaluate(ref, cand, bars: dict, *, group: str = "kind", cross_precision: bool = True) -> dict:
    """Compare `cand` with `ref` array by array against `bars` (as `derive_bars` gives them). An array whose group has no measured bar fails, and
    an array only one run holds is a problem: nothing is passed because it was not measured. Settings must match except dtype when
    `cross_precision` is true."""
    keys = tuple(k for k in ar.MATCH_KEYS if not (cross_precision and k == "dtype"))
    diff = ar.settings_problems(ref, cand, keys)
    if diff:
        raise NoiseFloorError("the runs are not comparable: " + "; ".join(diff))
    results, problems = [], []
    for key in sorted(set(ref.entries) | set(cand.entries)):
        kind, name = key
        if key not in ref.entries or key not in cand.entries:
            problems.append(f"{kind}/{name} is only in the {'reference' if key in ref.entries else 'candidate'}")
            continue
        g = "noise" if kind == "noise" else group_of(kind, name, group)
        m = ar.metrics(ref.array(kind, name), cand.array(kind, name))
        if g not in bars:
            ok, why = False, [f"no measured bar for group {g}"]
        else:
            ok, why = ar.verdict(m, bars[g])
        results.append({"kind": kind, "name": name, "group": g, "ok": ok, "reasons": why, **{k: m.get(k) for k in ("comparable", "pcc", "rel_l2", "slope")},
                        "bars": bars[g].__dict__ if g in bars else None})
    if not results:
        problems.append("the runs share no array")
    return {"ok": not problems and all(r["ok"] for r in results), "problems": problems, "results": results,
            "failing": [f"{r['kind']}/{r['name']}" for r in results if not r["ok"]]}


def detectable(good_eval: dict, bars: dict) -> dict:
    """Per group, the smallest extra relative L2 error (as a fraction of the array's norm) that moves an array of the good run past its bar, taking
    an independent error to add in quadrature to what the good run already has. 0 means the good run is at or past its bar already."""
    out = {}
    for r in good_eval["results"]:
        if not r.get("comparable") or r["bars"] is None or r["kind"] == "noise":
            continue
        q = math.sqrt(max(r["bars"]["rel_l2"] ** 2 - r["rel_l2"] ** 2, 0.0))
        cur = out.get(r["group"])
        if cur is None or q < cur["min_relative_error"]:
            out[r["group"]] = {"min_relative_error": q, "array": f"{r['kind']}/{r['name']}", "bar_rel_l2": r["bars"]["rel_l2"], "good_rel_l2": r["rel_l2"]}
    return out


# ---- the known-bad control --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Damage:
    """Which arrays of a good run to damage and how. mode `bf16` rounds each value to bfloat16 (a literal dtype error); mode `relative` adds an
    error of relative L2 size `size` (a fraction of the array's norm), drawn from `seed`."""
    arrays: tuple
    mode: str = "bf16"
    size: float | None = None
    seed: int = 0

    def label(self) -> str:
        what = ", ".join(f"{k}/{n}" for k, n in self.arrays)
        return f"damaged: {what} rounded to bfloat16" if self.mode == "bf16" else f"damaged: {what} given a relative error of {self.size}"


def damage_array(arr, mode: str, size, *, seed: int = 0) -> np.ndarray:
    a = np.ascontiguousarray(np.asarray(arr), dtype=np.float32)
    if not np.isfinite(a).all():
        raise NoiseFloorError("an array that is not finite cannot be damaged on purpose: it is already broken")
    if mode == "bf16":
        u = a.view(np.uint32).astype(np.uint64)
        u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000          # round to nearest even, keep the top 16 bits
        return u.astype(np.uint32).view(np.float32).reshape(a.shape)
    if mode == "relative":
        if not (_num(size) and size > 0):
            raise NoiseFloorError("a relative damage needs a size above 0")
        norm = float(np.linalg.norm(a.astype(np.float64)))
        d = np.random.default_rng(seed).standard_normal(a.shape)
        dn = float(np.linalg.norm(d))
        if norm == 0 or dn == 0:
            raise NoiseFloorError("an all-zero array has no relative size to damage")
        return (a + d * (size * norm / dn)).astype(np.float32)
    raise NoiseFloorError(f"unknown damage mode {mode!r}; use bf16 or relative")


def make_damaged_run(good, out_dir, damage: Damage) -> Path:
    """Write a new run directory that holds every array of `good`, with the arrays named in `damage` damaged and noted as such. `good` is not
    changed and `out_dir` must not exist."""
    if not damage.arrays:
        raise NoiseFloorError("the damage names no array")
    for key in damage.arrays:
        if tuple(key) not in good.entries:
            raise NoiseFloorError(f"the good run holds no {key[0]}/{key[1]} to damage")
    rec = ar.Recorder(out_dir, model=good.manifest["model"], library=good.manifest["library"], settings=good.settings,
                      seeds=good.manifest.get("seeds") or {})
    try:
        for (kind, name) in good.entries:
            arr = good.array(kind, name)
            if (kind, name) in {tuple(k) for k in damage.arrays}:
                rec.record(kind, name, damage_array(arr, damage.mode, damage.size, seed=damage.seed), note=f"damaged control: {damage.label()}")
            else:
                rec.record(kind, name, arr)
        return rec.finish().parent
    finally:
        rec.close()


def default_damage(good, mode: str = "bf16", size=None) -> Damage:
    """The highest-numbered block of kind `layer`, or else the last array of that kind."""
    stack = fd.Stack()
    blocks = sorted((n for (k, n) in good.entries if k == "layer" and stack.block_number(n) is not None), key=stack.block_number)
    if blocks:
        return Damage((("layer", blocks[-1]),), mode, size)
    layers = [n for (k, n) in good.entries if k == "layer"]
    if not layers:
        raise NoiseFloorError("the good run holds no array of kind layer to damage; name the arrays to damage")
    return Damage((("layer", layers[-1]),), mode, size)


# ---- the whole step -----------------------------------------------------------------------------------------------

def analyse(reference_env, chip_env, good, *, margin: float = DEFAULT_MARGIN, group: str = "kind", slope_rule: str = "floor", damage: Damage | None = None, bad_runs=(),
            work_dir=None, keep_work: bool = False) -> dict:
    """Measure the floor, set the bars and test them. `reference_env[0]` is the reference the good run is compared with. `work_dir` (a new directory)
    receives the damaged control run; without it a temporary directory is used and removed unless `keep_work`. See the module text."""
    floor = measure_floor(reference_env, chip_env, group=group)
    bars = derive_bars(floor, margin=margin, slope_rule=slope_rule)
    ref = list(reference_env)[0]
    good_eval = evaluate(ref, good, bars, group=group)
    damage = damage or default_damage(good)
    own_dir = work_dir is None
    base = Path(tempfile.mkdtemp(prefix="orchard-noise-floor-")) if own_dir else Path(work_dir)
    controls, reasons = [], []
    try:
        if not own_dir:
            try:
                base.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                raise NoiseFloorError(f"{base} exists; a work directory is never reused") from None
        damaged_path = make_damaged_run(good, base / "damaged", damage)
        with ar.load_run(damaged_path) as dmg:
            ev = evaluate(ref, dmg, bars, group=group)
        controls.append(_control(damage.label(), ev, good_eval, "synthetic", damage.arrays))
        for i, bad in enumerate(bad_runs):
            ev = evaluate(ref, bad, bars, group=group)
            controls.append(_control(f"recorded bad run {getattr(getattr(bad, 'root', None), 'name', None) or i}", ev, good_eval, "recorded", None))
    finally:
        if own_dir and not keep_work:
            shutil.rmtree(base, ignore_errors=True)
    det = detectable(good_eval, bars)
    if not good_eval["ok"]:
        reasons.append("the good run fails the bars it was meant to pass (" + ", ".join(good_eval["failing"][:6]) + "): a larger margin or a bar "
                       "that includes the chip's own arithmetic is needed")
    for c in controls:
        if c["detected"]:
            continue
        if c["kind"] == "synthetic":
            touched = [f"{k}/{n}" for k, n in damage.arrays]
            groups = {group_of(k, n, group) for k, n in damage.arrays}
            seen = ", ".join(f"{g}: {det[g]['min_relative_error']:.4f}" for g in sorted(groups) if g in det)
            reasons.append(f"{c['label']} is not detected: no damaged array fails where the good run passes. "
                           f"Extra relative error below about {seen or 'n/a'} passes unseen")
        else:
            reasons.append(f"{c['label']} passes the bars (or fails only where the good run fails too), so they cannot reject it")
    return {"schema": 1, "margin": margin, "group": group, "slope_rule": slope_rule, "floor": floor, "bars": {g: {**b.__dict__} for g, b in bars.items()},
            "good": {"ok": good_eval["ok"], "failing": good_eval["failing"], "problems": good_eval["problems"], "results": good_eval["results"]},
            "controls": controls, "detectable": det, "usable": not reasons, "reasons": reasons, "fallback": None}


def _control(label: str, ev: dict, good_eval: dict, kind: str, damaged) -> dict:
    """A control is detected only when it fails on an array that the good run passes. A control that fails where the good run fails already tells
    nothing: the bars reject everything there. A synthetic control must also fail on each array that was damaged."""
    new = [f for f in ev["failing"] if f not in good_eval["failing"]]
    if damaged is not None:
        detected = all(f"{k}/{n}" in new for k, n in damaged)
    else:
        detected = bool(new)
    return {"label": label, "ok": ev["ok"], "failing": ev["failing"], "newly_failing": new, "detected": detected, "kind": kind}


def _valid_bars(entry) -> ar.Bars:
    if not isinstance(entry, dict) or not all(_num(entry.get(k)) for k in ("pcc", "rel_l2", "slope_tol")):
        raise NoiseFloorError("a bar in the result is malformed")
    if entry["rel_l2"] < 0 or entry["slope_tol"] < 0 or not -1.0 <= entry["pcc"] <= 1.0:
        raise NoiseFloorError("a bar in the result is out of range")
    return ar.Bars(entry["pcc"], entry["rel_l2"], entry["slope_tol"], str(entry.get("source", ""))[:300])


def bars_or_fallback(result: dict, *, allow_fallback: bool = False) -> tuple[dict, str]:
    """The measured bars of an `analyse` result and a label for the report. When the result says the bars are unusable this raises, unless
    `allow_fallback` is true: then the provisional `audio_reference.DEFAULT_BARS` come back, labelled with the reason. The provisional bars are keyed by
    kind, so a result made with `group="family"` falls back to them by kind."""
    if not isinstance(result, dict) or not isinstance(result.get("usable"), bool) or not isinstance(result.get("reasons"), list):
        raise NoiseFloorError("the noise-floor result is malformed")
    if not result["usable"]:
        why = "; ".join(str(r) for r in result["reasons"])[:600]
        if not allow_fallback:
            raise NoiseFloorError(f"the measured bars are unusable ({why}); ask for the fallback explicitly to use the provisional bars")
        return dict(ar.DEFAULT_BARS), f"fallback: provisional audio_reference.DEFAULT_BARS (a choice that was never measured), because the measured bars are unusable: {why}"
    raw = result.get("bars")
    if not isinstance(raw, dict) or not raw:
        raise NoiseFloorError("the noise-floor result holds no bars")
    bars = {g: _valid_bars(e) for g, e in raw.items()}
    margin = result.get("margin")
    if not _num(margin):
        raise NoiseFloorError("the noise-floor result holds no margin")
    return bars, f"measured: {margin} times the worst CPU-environment spread (slope rule {result.get('slope_rule', 'floor')}), checked against a known-bad control"


# ---- reading the result ----------------------------------------------------------------------------------------

def summarise(res: dict) -> list[str]:
    runs = res["floor"]["runs"]
    lines = [f"Floor from {len(runs['reference_environment'])} runs in the reference environment and {len(runs['chip_environment'])} in the chip's environment; "
             f"margin {res['margin']}; bars grouped by {res['group']}; slope rule {res['slope_rule']}."]
    for g, info in res["floor"]["groups"].items():
        w, b = info["worst"], res["bars"][g]
        lines.append(f"  {g}: floor relative L2 {w['rel_l2']:.5f} (worst array {info['worst_array']}), pcc {w['pcc']:.5f}; "
                     f"bar relative L2 {b['rel_l2']:.5f}, pcc {b['pcc']:.5f}, slope within {b['slope_tol']:.5f} of 1")
    g = res["good"]
    if g["ok"]:
        lines.append("The good run passes these bars.")
    else:
        first = next((r for r in g["results"] if not r["ok"]), None)
        lines.append(f"The good run fails these bars on {len(g['failing'])} arrays, among them {', '.join(g['failing'][:4])}"
                     + (f" ({first['reasons'][0]})." if first else "."))
    for c in res["controls"]:
        lines.append(f"Control ({c['label']}): " + ("fails the bars where the good run passes, as it should" if c["detected"]
                                                      else "is not told apart from the good run by these bars"))
    for gname, d in res["detectable"].items():
        lines.append(f"  {gname}: extra relative error below {d['min_relative_error']:.4f} passes unseen (checked on {d['array']})")
    lines.append("Result: the bars are usable." if res["usable"] else "Result: the bars are unusable.")
    for r in res["reasons"]:
        lines.append("  Reason: " + r)
    return lines


# ---- command line --------------------------------------------------------------------------------------------------

def main(argv=None, out=None) -> int:
    import argparse
    import sys
    from orchard import safeio
    out = out or sys.stdout
    p = argparse.ArgumentParser(prog="python -m orchard.noise_floor", description=__doc__.split("\n\n")[0])
    p.add_argument("--ref-env", nargs="+", required=True, help="CPU run directories in the reference environment; the first is the reference")
    p.add_argument("--chip-env", nargs="*", default=[], help="CPU run directories of the same code in the chip's own environment")
    p.add_argument("--good", required=True, help="run directory of a candidate that should pass (for example a good chip run)")
    p.add_argument("--bad-run", action="append", default=[], help="a recorded run that must fail the bars (repeatable)")
    p.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    p.add_argument("--group", choices=GROUPS, default="kind")
    p.add_argument("--slope-rule", choices=SLOPE_RULES, default="floor", help="slope bar from the CPU slope floor, or equal to the relative L2 bar (see derive_bars)")
    p.add_argument("--damage", action="append", help="kind/name of an array of the good run to damage (repeatable; default: the last layer block)")
    p.add_argument("--damage-mode", choices=("bf16", "relative"), default="bf16")
    p.add_argument("--damage-size", type=float, help="relative L2 size for --damage-mode relative")
    p.add_argument("--work-dir", help="new directory that keeps the damaged run (default: a temporary directory that is removed)")
    p.add_argument("--json", action="store_true", help="print the full JSON result and no summary")
    p.add_argument("--out", help="also write the JSON result to this new file (an existing file is refused)")
    args = p.parse_args(argv)
    runs = []
    try:
        for d in args.ref_env + args.chip_env + [args.good] + args.bad_run:
            runs.append(ar.load_run(d))
        n_ref, n_chip = len(args.ref_env), len(args.chip_env)
        ref_runs, chip_runs, good, bad = runs[:n_ref], runs[n_ref:n_ref + n_chip], runs[n_ref + n_chip], runs[n_ref + n_chip + 1:]
        damage = None
        if args.damage:
            pairs = []
            for item in args.damage:
                if item.count("/") != 1 or not all(item.split("/")):
                    raise NoiseFloorError(f"--damage must be kind/name; got {item!r}")
                pairs.append(tuple(item.split("/")))
            damage = Damage(tuple(pairs), args.damage_mode, args.damage_size)
        elif args.damage_mode == "relative":
            damage = default_damage(good, "relative", args.damage_size)
        res = analyse(ref_runs, chip_runs, good, margin=args.margin, group=args.group, slope_rule=args.slope_rule, damage=damage, bad_runs=bad, work_dir=args.work_dir)
        text = fd.to_json(res)
        if args.out:
            try:
                safeio.atomic_write(args.out, text + "\n", replace=False)
            except (OSError, safeio.SafeReadError) as exc:
                raise NoiseFloorError(f"cannot write {args.out}: {exc}") from exc
    except (ReferenceError, NoiseFloorError) as exc:
        print(f"refused: {exc}", file=out)
        return 2
    finally:
        for r in runs:
            r.close()
    if args.json:
        print(text, file=out)
    else:
        for line in summarise(res):
            print(line, file=out)
    return 0 if res["usable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
