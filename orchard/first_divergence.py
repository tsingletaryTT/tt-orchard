# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""First-divergence probe (issue 172): find the first block of a layer stack, and the first operation inside that block, where a candidate run
(for example the chip) leaves a stated tolerance around a reference run (for example the CPU).

Both runs are recorded runs in the layout of `orchard/audio_reference.py` (a manifest and one `.npy` per array, read through `audio_reference.load_run`,
which pins the directory, caps sizes and checks every hash). This module adds no reader of its own and does not depend on any one model's chip
script. A stage can call `find_first_divergence(ref, cand, ...)` with two loaded runs; a person can run
`python -m orchard.first_divergence REF CAND ...`.

Naming. A block output is an array of kind `layer` (set by `Stack.kind`) whose name matches `Stack.block_re` (default `layer_<n>`). An operation inside block
`layer_05` is an array of the same kind named `layer_05.<op>` and holds the stream after that operation. Operations are taken in the order the reference
manifest lists them, or in the order given in `ops_order`. A probe that reads back each section of one layer records exactly these names.

What it reports, for every block, every operation of the block it located, and every other array both runs hold:
  * PCC, relative L2 error and slope (the slope exposes a scaled output that PCC passes), the largest absolute error;
  * the worst rows and the worst channels (the last axis is the channel; all other axes are flattened into rows);
  * the high-norm rows reported apart: rows of the reference whose norm is more than `high_norm_factor` times the median row norm. It gives their
    relative error, the error of the other rows, the share of the squared error they carry and the share of the reference energy they hold. A few
    large rows can carry most of an error that the whole-tensor number averages away (issue 163);
  * the environment floor next to each number: the same numbers for two CPU runs in different environments (`floor=(a, b)`), so a reader can tell
    drift from noise. With no stated tolerance the tolerance of each tensor is `margin` times its floor.

Bisect. Errors normally grow along a stack, so the first block that leaves the tolerance is found by halving (`bisect.block_probes` lists which blocks were
read). With `table=True` (the default) every block is also compared, the first block that leaves is taken from that scan, and the report says when
the two methods disagree: errors that fall back inside the tolerance later (a layer that cancels an earlier error) make a bisect miss the first block.
The operations of the located block are searched the same way. The stream after an operation includes the error of every earlier operation, so the
first operation that leaves is where the error entered, to the resolution of the recorded operations.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

import numpy as np

from orchard import audio_reference as ar

ReferenceError = ar.ReferenceError


class DivergenceError(Exception):
    """The two runs cannot be compared, or an option is unusable."""


FALLBACK_REL_L2 = 0.05          # choice: used only when there is neither a stated tolerance nor a floor for a tensor
DEFAULT_MARGIN = 3.0            # choice: a tensor leaves its tolerance above this many times its environment floor
MIN_FLOOR = 1e-6                # choice: a floor of zero (two environments that agree exactly) is lifted to this
HIGH_NORM_FACTOR = 2.0          # choice: a row is high-norm above this many times the median non-zero row norm


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


@dataclass(frozen=True)
class Tolerance:
    """When a tensor has left the reference. `rel_l2` is a stated bound on the relative L2 error; when None, each tensor's bound is `margin` times
    its own environment floor, and `fallback_rel_l2` where it has none. `min_pcc` is an optional second bound."""
    rel_l2: float | None = None
    margin: float = DEFAULT_MARGIN
    fallback_rel_l2: float = FALLBACK_REL_L2
    min_pcc: float | None = None

    def __post_init__(self):
        if self.rel_l2 is not None and not (_num(self.rel_l2) and self.rel_l2 >= 0):
            raise DivergenceError("the tolerance rel_l2 must be a finite number of at least 0")
        if not (_num(self.margin) and self.margin >= 1):
            raise DivergenceError("the margin must be a finite number of at least 1")
        if not (_num(self.fallback_rel_l2) and self.fallback_rel_l2 > 0):
            raise DivergenceError("the fallback tolerance must be a finite number above 0")
        if self.min_pcc is not None and not (_num(self.min_pcc) and -1 <= self.min_pcc <= 1):
            raise DivergenceError("min_pcc must be a number between -1 and 1")


@dataclass(frozen=True)
class Stack:
    """Where the blocks of the stack and their operations are in a run."""
    kind: str = "layer"
    block_re: str = r"layer_(\d+)"
    op_sep: str = "."

    def __post_init__(self):
        try:
            pat = re.compile(self.block_re)
        except re.error as exc:
            raise DivergenceError(f"block_re is not a pattern: {exc}") from exc
        if pat.groups < 1:
            raise DivergenceError("block_re needs one group that holds the block number")
        if self.kind not in ar.KINDS:
            raise DivergenceError(f"unknown kind {self.kind!r}; allowed: {ar.KINDS}")
        if not self.op_sep:
            raise DivergenceError("op_sep must not be empty")

    def block_number(self, name: str):
        m = re.fullmatch(self.block_re, name)
        if not m:
            return None
        try:
            return int(m.group(1))
        except ValueError:
            return None


class MemoryRun:
    """Arrays held in memory, with the two members of a loaded run that this module uses: `entries` (kind, name keys, in recorded order) and
    `array(kind, name)`. For a stage that has the arrays already, and for tests."""

    def __init__(self, arrays: dict, settings: dict | None = None):
        self._arrays = {k: np.asarray(v) for k, v in arrays.items()}
        self.entries = {k: {"kind": k[0], "name": k[1]} for k in self._arrays}
        if settings is not None:
            self.settings = settings

    def array(self, kind, name):
        return self._arrays[(kind, name)]


# ---- the numbers for one tensor ----------------------------------------------------------------------------------

def _rows(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return a.reshape(1, -1) if a.ndim <= 1 else a.reshape(-1, a.shape[-1])


def tensor_report(ref, cand, *, top: int = 5, high_norm_factor: float = HIGH_NORM_FACTOR, row_groups: dict | None = None) -> dict:
    """PCC, relative L2, slope and the largest error of `cand` against `ref`, with the worst rows, the worst channels and the high-norm rows apart.
    A pair that cannot be compared (different shapes, a value that is not finite) is returned with `comparable` false and the reason."""
    ref, cand = np.asarray(ref), np.asarray(cand)
    m = ar.metrics(ref, cand)
    out = {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in m.items()}
    out.setdefault("pcc", None)
    out["error_mostly_in_high_norm_rows"] = False
    out["high_norm"] = None
    out["row_groups"] = []
    out["worst_rows"], out["worst_channels"] = [], []
    if not m.get("comparable"):
        return out
    out["shape"] = list(ref.shape)
    x, y = _rows(ref), _rows(cand)
    err = y - x
    row_err = np.linalg.norm(err, axis=1)
    row_ref = np.linalg.norm(x, axis=1)
    col_err = np.linalg.norm(err, axis=0)
    col_ref = np.linalg.norm(x, axis=0)

    def worst(errs, refs, label):
        order = np.argsort(-errs, kind="stable")[:max(0, int(top))]
        return [{label: int(i), "err_norm": float(errs[i]), "ref_norm": float(refs[i]),
                 "rel_err": (float(errs[i] / refs[i]) if refs[i] > 0 else (0.0 if errs[i] == 0 else None))} for i in order if errs[i] > 0]
    out["worst_rows"] = worst(row_err, row_ref, "row")
    out["worst_channels"] = worst(col_err, col_ref, "channel")
    sq_err_all, sq_ref_all = float((err ** 2).sum()), float((x ** 2).sum())
    for gname, (start, stop) in (row_groups or {}).items():
        if not (0 <= start < stop <= row_ref.size):
            continue                                   # the groups name rows of a block's layout; an array with fewer rows has none of them
        g_err, g_ref = float((err[start:stop] ** 2).sum()), float((x[start:stop] ** 2).sum())
        out["row_groups"].append({"name": gname, "rows": [start, stop], "rel_l2": math.sqrt(g_err / g_ref) if g_ref > 0 else None,
                                  "error_share": (g_err / sq_err_all) if sq_err_all > 0 else 0.0,
                                  "energy_share": (g_ref / sq_ref_all) if sq_ref_all > 0 else None})
    nonzero = row_ref[row_ref > 0]
    if nonzero.size:
        threshold = high_norm_factor * float(np.median(nonzero))
        hi = row_ref > threshold
        n_hi = int(hi.sum())
        if 0 < n_hi < row_ref.size:
            sq_err, sq_ref = float((err ** 2).sum()), float((x ** 2).sum())
            hi_err, hi_ref = float((err[hi] ** 2).sum()), float((x[hi] ** 2).sum())
            rest_err, rest_ref = sq_err - hi_err, sq_ref - hi_ref
            share = (hi_err / sq_err) if sq_err > 0 else 0.0
            out["high_norm"] = {
                "rows": n_hi, "of": int(row_ref.size), "threshold_norm": threshold, "indices": [int(i) for i in np.flatnonzero(hi)[:8]],
                "rel_l2": math.sqrt(hi_err / hi_ref) if hi_ref > 0 else None,
                "rest_rel_l2": math.sqrt(max(rest_err, 0.0) / rest_ref) if rest_ref > 0 else None,
                "error_share": share, "energy_share": (hi_ref / sq_ref) if sq_ref > 0 else None}
            out["error_mostly_in_high_norm_rows"] = bool(share >= 0.5)
    return out


# ---- the search ----------------------------------------------------------------------------------------------------

def first_leaving(n: int, leaves) -> tuple[int | None, list[int]]:
    """Bisect over `n` ordered items for the first one for which `leaves(i)` is true, assuming that once an item leaves, the later ones do too.
    Returns the index (None when the last item does not leave) and the indices that were asked, in order."""
    probes: list[int] = []
    if n <= 0:
        return None, probes
    probes.append(n - 1)
    if not leaves(n - 1):
        return None, probes
    lo, hi = -1, n - 1                  # lo is within tolerance (the input of the stack), hi has left
    while hi - lo > 1:
        mid = (lo + hi) // 2
        probes.append(mid)
        if leaves(mid):
            hi = mid
        else:
            lo = mid
    return hi, probes


class _Judge:
    """Compares one array of the two runs, caches the result, and decides whether it leaves its tolerance."""

    def __init__(self, ref, cand, floor, tol: Tolerance, top: int, factor: float, stack: Stack, row_groups: dict | None):
        self.ref, self.cand, self.floor, self.tol, self.top, self.factor = ref, cand, floor, tol, top, factor
        self.stack, self.row_groups = stack, row_groups
        self._cache: dict = {}

    def entry(self, kind: str, name: str) -> dict:
        key = (kind, name)
        if key in self._cache:
            return self._cache[key]
        try:
            in_stack = kind == self.stack.kind and self.stack.block_number(name.split(self.stack.op_sep, 1)[0]) is not None
            e = tensor_report(self.ref.array(kind, name), self.cand.array(kind, name), top=self.top, high_norm_factor=self.factor,
                              row_groups=self.row_groups if in_stack else None)
        except ReferenceError as exc:
            e = {"comparable": False, "reason": f"unreadable: {exc}", "worst_rows": [], "worst_channels": [], "high_norm": None,
                 "error_mostly_in_high_norm_rows": False, "pcc": None, "row_groups": []}
        e = {"kind": kind, "name": name, **e}
        e["floor"] = self._floor(kind, name)
        floor_l2 = e["floor"]["rel_l2"] if e["floor"] else None
        if self.tol.rel_l2 is not None:
            e["tolerance"], e["tolerance_source"] = self.tol.rel_l2, "stated"
        elif floor_l2 is not None:
            e["tolerance"] = self.tol.margin * max(floor_l2, MIN_FLOOR)
            e["tolerance_source"] = (f"{self.tol.margin} times the environment floor" if floor_l2 >= MIN_FLOOR
                                     else f"{self.tol.margin} times the smallest floor {MIN_FLOOR} (the two environments agree exactly)")
        else:
            e["tolerance"], e["tolerance_source"] = self.tol.fallback_rel_l2, "fallback (no environment floor)"
        e["ratio_to_floor"] = (e["rel_l2"] / floor_l2) if (e.get("rel_l2") is not None and floor_l2) else None
        why_leaves = []
        if not e.get("comparable"):
            why_leaves.append(e.get("reason", "not comparable"))
        else:
            if e["rel_l2"] is None or e["rel_l2"] > e["tolerance"]:
                why_leaves.append(f"relative L2 {e['rel_l2']} is above {e['tolerance']}")
            if self.tol.min_pcc is not None and (e["pcc"] is None or e["pcc"] < self.tol.min_pcc):
                why_leaves.append(f"pcc {e['pcc']} is below {self.tol.min_pcc}")
        e["leaves"], e["leaves_because"] = bool(why_leaves), why_leaves
        self._cache[key] = e
        return e

    def _floor(self, kind, name):
        if self.floor is None:
            return None
        a, b = self.floor
        if (kind, name) not in a.entries or (kind, name) not in b.entries:
            return None
        try:
            m = ar.metrics(a.array(kind, name), b.array(kind, name))
        except ReferenceError:
            return None
        if not m.get("comparable") or m.get("pcc") is None:
            return None
        return {"rel_l2": m["rel_l2"], "pcc": m["pcc"], "slope": m["slope"]}

    def leaves(self, kind, name) -> bool:
        return self.entry(kind, name)["leaves"]


def _settings_problems(a, b, label, *, cross_precision: bool) -> list[str]:
    sa, sb = getattr(a, "settings", None), getattr(b, "settings", None)
    if sa is None or sb is None:
        return []
    keys = tuple(k for k in ar.MATCH_KEYS if not (cross_precision and k == "dtype"))
    return [f"{label}: settings.{k} differs: {sa.get(k)!r} and {sb.get(k)!r}" for k in keys if sa.get(k) != sb.get(k)]


def find_first_divergence(ref, cand, *, stack: Stack | None = None, tolerance: Tolerance | None = None, floor=None, top: int = 5,
                          high_norm_factor: float = HIGH_NORM_FACTOR, table: bool = True, ops_order: dict | None = None,
                          cross_precision: bool = True, row_groups: dict | None = None) -> dict:
    """Compare `cand` with `ref` and locate the first block, then the first operation, that leaves the tolerance. See the module text.

    `floor` is a pair of runs of the same code in two environments (for example the CPU reference and the CPU code in the chip's own Python
    environment); their agreement is the environment floor printed next to each number. The two runs of the pair must match each other fully
    and match `ref` except for dtype. `row_groups` names row ranges of a block's layout, {"memory": (0, 64)}: for each block and operation the
    report gives the relative error, the share of the squared error and the share of the reference energy of those rows. `cross_precision` leaves dtype out of the check of `ref` against `cand`, since a chip run differs in
    dtype from a float32 reference. Raises `DivergenceError` when the runs do not belong together."""
    stack, tol = stack or Stack(), tolerance or Tolerance()
    problems = _settings_problems(ref, cand, "candidate", cross_precision=cross_precision)
    if floor is not None:
        if not (isinstance(floor, (tuple, list)) and len(floor) == 2):
            raise DivergenceError("floor must be a pair of runs")
        problems += _settings_problems(floor[0], floor[1], "floor pair", cross_precision=False)
        problems += _settings_problems(ref, floor[0], "floor run", cross_precision=True)
    if problems:
        raise DivergenceError("the runs are not comparable: " + "; ".join(problems))
    for gname, span in (row_groups or {}).items():
        if not (isinstance(gname, str) and isinstance(span, (tuple, list)) and len(span) == 2
                and all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in span) and span[0] < span[1]):
            raise DivergenceError(f"row group {gname!r} must be (start, stop) with 0 <= start < stop")
    judge = _Judge(ref, cand, floor, tol, top, high_norm_factor, stack, {k: tuple(v) for k, v in (row_groups or {}).items()} or None)

    in_ref = [n for (k, n) in ref.entries if k == stack.kind and stack.block_number(n) is not None]
    if not in_ref:
        raise DivergenceError(f"the reference holds no block of kind {stack.kind!r} matching {stack.block_re!r}")
    in_ref.sort(key=stack.block_number)
    problems_out, warnings, notes = [], [], []
    blocks = []
    for n in in_ref:
        if (stack.kind, n) in cand.entries:
            blocks.append(n)
        else:
            problems_out.append(f"{stack.kind}/{n} is in the reference and not in the candidate")
    extra = sorted(n for (k, n) in cand.entries if k == stack.kind and stack.block_number(n) is not None and n not in in_ref)
    problems_out += [f"{stack.kind}/{n} is in the candidate and not in the reference" for n in extra]
    if not blocks:
        raise DivergenceError("the candidate holds none of the reference's blocks")

    def block_leaves(i):
        return judge.leaves(stack.kind, blocks[i])
    bisect_idx, block_probes = first_leaving(len(blocks), block_leaves)
    scan_idx = None
    if table:
        flags = [block_leaves(i) for i in range(len(blocks))]
        scan_idx = flags.index(True) if True in flags else None
        if scan_idx != bisect_idx:
            warnings.append(f"errors along the stack are not monotone: a scan of every block first leaves the tolerance at "
                            f"{blocks[scan_idx] if scan_idx is not None else 'no block'}, a bisect gives "
                            f"{blocks[bisect_idx] if bisect_idx is not None else 'no block'}; the scan is used")
        if scan_idx is not None and any(not f for f in flags[scan_idx:]):
            back = [blocks[i] for i in range(scan_idx, len(blocks)) if not flags[i]]
            warnings.append("errors along the stack are not monotone: " + ", ".join(back[:5]) + " is back within the tolerance after "
                            f"{blocks[scan_idx]} left it")
    chosen = scan_idx if table else bisect_idx
    first_block = judge.entry(stack.kind, blocks[chosen]) if chosen is not None else None

    ops, first_op, op_probes = [], None, []
    if first_block is not None:
        prefix = blocks[chosen] + stack.op_sep
        op_names = [n for (k, n) in ref.entries if k == stack.kind and n.startswith(prefix) and n != blocks[chosen]]
        if ops_order and blocks[chosen] in ops_order:
            wanted = [prefix + o for o in ops_order[blocks[chosen]]]
            missing = [w for w in wanted if w not in op_names]
            if missing:
                raise DivergenceError(f"ops_order names {missing}, which the reference does not hold")
            op_names = wanted
        op_names = [n for n in op_names if (stack.kind, n) in cand.entries]
        if not op_names:
            notes.append(f"no operations recorded for {blocks[chosen]} in both runs, so the search stops at the block")
        else:
            idx, op_probes = first_leaving(len(op_names), lambda i: judge.leaves(stack.kind, op_names[i]))
            if table or idx is None:
                flags = [judge.leaves(stack.kind, n) for n in op_names]
                idx = flags.index(True) if True in flags else None
            ops = [judge.entry(stack.kind, n) for n in (op_names if table else [op_names[i] for i in sorted(set(op_probes))])]
            if idx is None:
                notes.append(f"no recorded operation of {blocks[chosen]} leaves the tolerance although the block output does: the error enters "
                             "between recorded operations, or on the block's output")
            else:
                first_op = judge.entry(stack.kind, op_names[idx])

    def in_stack(kind, name):
        return kind == stack.kind and stack.block_number(name.split(stack.op_sep, 1)[0]) is not None
    other = []
    for key in ref.entries:
        if in_stack(*key):
            continue
        if key not in cand.entries:
            problems_out.append(f"{key[0]}/{key[1]} is in the reference and not in the candidate")
            continue
        other.append(judge.entry(*key))
    return {
        "located": first_block is not None,
        "complete": not problems_out,
        "stack": {"kind": stack.kind, "block_re": stack.block_re, "blocks": len(blocks)},
        "tolerance": {"rel_l2": tol.rel_l2, "margin": tol.margin, "fallback_rel_l2": tol.fallback_rel_l2, "min_pcc": tol.min_pcc},
        "environment_floor": "recorded" if floor is not None else "none given",
        "first_block": first_block, "first_op": first_op,
        "bisect": {"block_probes": [blocks[i] for i in block_probes], "op_probes": [op_names[i] for i in op_probes] if first_block is not None and ops else []},
        "bisect_agrees_with_scan": (bisect_idx == scan_idx) if table else None,
        "blocks": [judge.entry(stack.kind, n) for n in blocks] if table else [judge.entry(stack.kind, blocks[i]) for i in sorted(set(block_probes))],
        "ops": ops, "tensors": other, "problems": problems_out, "warnings": warnings, "notes": notes}


# ---- reading the report ----------------------------------------------------------------------------------------

def _f(x, digits=4):
    return "n/a" if x is None else f"{x:.{digits}f}"


def _entry_line(e: dict) -> str:
    if not e.get("comparable"):
        return f"{e['kind']}/{e['name']}: cannot be compared ({e.get('reason')})"
    line = f"{e['kind']}/{e['name']}: pcc {_f(e['pcc'], 5)}, relative L2 {_f(e['rel_l2'])}, slope {_f(e['slope'])}"
    fl = e.get("floor")
    if fl:
        line += f"; environment floor: relative L2 {_f(fl['rel_l2'])}, pcc {_f(fl['pcc'], 5)}"
        if e.get("ratio_to_floor") is not None:
            line += f" (this run is {e['ratio_to_floor']:.1f} times the floor)"
    else:
        line += "; no environment floor"
    return line + f"; tolerance {_f(e['tolerance'])} ({e['tolerance_source']})"


def _rows_line(e: dict) -> list[str]:
    out = []
    if e.get("worst_rows"):
        out.append("    worst rows: " + ", ".join(f"row {r['row']} (error {r['err_norm']:.3g} of norm {r['ref_norm']:.3g})" for r in e["worst_rows"][:3]))
    if e.get("worst_channels"):
        out.append("    worst channels: " + ", ".join(f"channel {c['channel']} (error {c['err_norm']:.3g})" for c in e["worst_channels"][:3]))
    for g in e.get("row_groups") or []:
        out.append(f"    rows {g['name']} ({g['rows'][0]} to {g['rows'][1] - 1}): relative L2 {_f(g['rel_l2'])}, "
                   f"{_f(100 * g['error_share'], 1)}% of the squared error, {_f(None if g['energy_share'] is None else 100 * g['energy_share'], 1)}% of the reference energy")
    hn = e.get("high_norm")
    if hn:
        out.append(f"    {hn['rows']} high-norm rows of {hn['of']} ({_f(None if hn['energy_share'] is None else 100 * hn['energy_share'], 1)}% of the reference energy) "
                   f"carry {_f(100 * hn['error_share'], 1)}% of the squared error; their relative L2 is {_f(hn['rel_l2'])} against {_f(hn['rest_rel_l2'])} for the other rows")
    return out


def summarise(rep: dict) -> list[str]:
    """A short plain-English account of a report."""
    lines = []
    s = rep["stack"]
    if not rep["located"]:
        lines.append(f"All {s['blocks']} blocks of kind {s['kind']} stay within their tolerance.")
    else:
        b = rep["first_block"]
        lines.append(f"The first block that leaves its tolerance is {b['name']}.")
        lines.append("  " + _entry_line(b))
        lines += _rows_line(b)
        op = rep["first_op"]
        if op is not None:
            lines.append(f"Inside it, the first operation that leaves is {op['name']}.")
            lines.append("  " + _entry_line(op))
            lines += _rows_line(op)
        for n in rep["notes"]:
            lines.append("Note: " + n)
    if rep["environment_floor"] == "none given":
        lines.append("There is no environment floor in this report: a reader cannot tell drift from noise.")
    for w in rep["warnings"]:
        lines.append("Warning: " + w)
    for p in rep["problems"]:
        lines.append("Problem: " + p)
    others = rep["tensors"]
    leaving = [t for t in others if t["leaves"]]
    lines.append(f"Other arrays: {len(others)} compared, {len(leaving)} leave their tolerance.")
    for t in leaving[:10]:
        lines.append("  " + _entry_line(t))
    if len(leaving) > 10:
        lines.append(f"  and {len(leaving) - 10} more (see the JSON report)")
    return lines


def _clean(obj):
    """Replace non-finite numbers by None so the report is strict JSON."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return _clean(obj.item())
    return obj


def to_json(rep: dict) -> str:
    return json.dumps(_clean(rep), indent=1, allow_nan=False, sort_keys=False)


# ---- command line --------------------------------------------------------------------------------------------------

def main(argv=None, out=None) -> int:
    import argparse
    import sys
    from orchard import safeio
    out = out or sys.stdout
    p = argparse.ArgumentParser(prog="python -m orchard.first_divergence", description=__doc__.split("\n\n")[0])
    p.add_argument("reference", help="run directory of the reference (for example the CPU run)")
    p.add_argument("candidate", help="run directory of the run under test (for example the chip run)")
    p.add_argument("--floor-ref", help="environment floor: one CPU run (usually the reference itself)")
    p.add_argument("--floor-other", help="environment floor: the same CPU code in the other environment")
    p.add_argument("--kind", default="layer")
    p.add_argument("--block-pattern", default=r"layer_(\d+)", help="regular expression for a block's array name; one group holds the number")
    p.add_argument("--rel-l2", type=float, help="stated tolerance on the relative L2 error (otherwise MARGIN times each tensor's floor)")
    p.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    p.add_argument("--fallback-rel-l2", type=float, default=FALLBACK_REL_L2)
    p.add_argument("--min-pcc", type=float)
    p.add_argument("--high-norm-factor", type=float, default=HIGH_NORM_FACTOR)
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--rows", action="append", default=[], metavar="NAME=START:STOP", help="a named row range of a block's layout, for example memory=0:64 (repeatable)")
    p.add_argument("--bisect-only", action="store_true", help="read only the blocks the bisect needs (no table; cannot see errors that come back inside)")
    p.add_argument("--strict-dtype", action="store_true", help="require the dtype setting to match too")
    p.add_argument("--json", action="store_true", help="print the full JSON report and no summary")
    p.add_argument("--out", help="also write the JSON report to this new file (an existing file is refused)")
    args = p.parse_args(argv)
    if bool(args.floor_ref) != bool(args.floor_other):
        print("refused: give both --floor-ref and --floor-other, or neither", file=out)
        return 2
    runs = []
    try:
        for d in (args.reference, args.candidate) + ((args.floor_ref, args.floor_other) if args.floor_ref else ()):
            runs.append(ar.load_run(d))
        ref, cand = runs[0], runs[1]
        floor = (runs[2], runs[3]) if args.floor_ref else None
        groups = {}
        for item in args.rows:
            m = re.fullmatch(r"([A-Za-z0-9_.\-]{1,40})=([0-9]{1,9}):([0-9]{1,9})", item)
            if not m:
                raise DivergenceError(f"--rows must be NAME=START:STOP, got {item!r}")
            groups[m.group(1)] = (int(m.group(2)), int(m.group(3)))
        rep = find_first_divergence(ref, cand, stack=Stack(kind=args.kind, block_re=args.block_pattern),
                                    tolerance=Tolerance(rel_l2=args.rel_l2, margin=args.margin, fallback_rel_l2=args.fallback_rel_l2, min_pcc=args.min_pcc),
                                    floor=floor, top=args.top, high_norm_factor=args.high_norm_factor, table=not args.bisect_only,
                                    cross_precision=not args.strict_dtype, row_groups=groups or None)
        text = to_json(rep)
        if args.out:
            try:
                safeio.atomic_write(args.out, text + "\n", replace=False)
            except (OSError, safeio.SafeReadError) as exc:
                raise DivergenceError(f"cannot write {args.out}: {exc}") from exc
    except (ReferenceError, DivergenceError) as exc:
        print(f"refused: {exc}", file=out)
        return 2
    finally:
        for r in runs:
            r.close()
    if args.json:
        print(text, file=out)
    else:
        for line in summarise(rep):
            print(line, file=out)
    return 1 if rep["located"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
