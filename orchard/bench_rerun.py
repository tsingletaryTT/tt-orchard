# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Re-run only the items whose reply was cut off, then merge the new results into the first run's.

    python3 -m orchard.bench_rerun items --run RUN/pi-default --phases humaneval,mbpp,ifeval --out items.json
    python3 -m orchard.bench_rerun merge --first RUN/pi-default --again RUN/pi-rerun --phases humaneval,mbpp,ifeval --out merged.json

`items` reads each phase's lm-evaluation-harness samples and lists the documents whose reply is empty. With thinking on, a reply that spends the whole
generation limit reasoning comes back with empty content (`finish_reason` length), so an empty reply marks a cut-off. The list is not exactly the proxy's cut-off
count: a reply cut off after its answer began has text, and a model can answer with nothing. The proxy records requests without document ids, so the
two cannot be matched item by item. The output is the file `orchard.bench_session --items` takes.

`merge` (with `--items`, the file the re-run was asked for) replaces the first run's result for each re-run item and recomputes each per-item metric over every item. It reads files and writes only its
`--out`. It refuses a re-run item the first run did not have, a second row for one item, a changed prompt or document hash, a re-run that is not exactly the requested items, and a sample missing a metric.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def samples_file(run_dir, phase: str) -> Path:
    found = sorted((Path(run_dir) / phase / "results").rglob("samples_*.jsonl"), key=lambda p: p.name)
    if not found:
        raise FileNotFoundError(f"no samples_*.jsonl for phase {phase} under {Path(run_dir) / phase / 'results'}")
    return found[-1]


def read_samples(run_dir, phase: str) -> list[dict]:
    rows = []
    # Split on "\n" only. str.splitlines() also cuts at U+2028, U+2029 and U+0085, which lm-eval leaves raw inside a JSON string.
    for line in samples_file(run_dir, phase).read_text(encoding="utf-8").split("\n"):
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _empty(sample: dict) -> bool:
    try:
        text = sample["resps"][0][0]
    except (KeyError, IndexError, TypeError):
        return False
    return isinstance(text, str) and not text.strip()


def cutoff_items(run_dir, phases) -> dict:
    """{phase: sorted document ids with an empty reply}; a phase with none is left out."""
    out = {}
    for phase in phases:
        ids = sorted({s["doc_id"] for s in read_samples(run_dir, phase) if _empty(s)})
        if ids:
            out[phase] = ids
    return out


def _metrics(sample: dict) -> dict:
    return {k: float(v) for k, v in sample.items() if k != "doc_id" and isinstance(v, (bool, int, float))}


def _by_doc(run_dir, phase: str, which: str) -> dict:
    """The phase's samples keyed by document id. A second row for one document (lm-eval writes one per filter) would be lost by the key, so it is an error."""
    rows = {}
    for s in read_samples(run_dir, phase):
        if s["doc_id"] in rows:
            raise ValueError(f"{phase}: the {which} has more than one row for item {s['doc_id']} (a task with several filters?); merge keys rows by item only")
        rows[s["doc_id"]] = s
    return rows


def _check_same_question(phase: str, first: dict, again: dict) -> None:
    for i, s in again.items():
        for key in ("doc_hash", "prompt_hash"):
            if key in s and key in first[i] and s[key] != first[i][key]:
                raise ValueError(f"{phase}: item {i} was asked differently in the re-run ({key} differs)")


def _check_requested(phase: str, again: dict, expected: dict) -> None:
    if phase not in expected:
        raise ValueError(f"{phase}: not in the requested items")
    want, got = set(expected[phase]), set(again)
    if want - got:
        raise ValueError(f"{phase}: the re-run is missing requested items {sorted(want - got)}")
    if got - want:
        raise ValueError(f"{phase}: the re-run has items that were not requested {sorted(got - want)}")


def _numbers(phase: str, rows: dict, names: list, which: str) -> dict:
    out = {}
    for i, s in rows.items():
        m = _metrics(s)
        for name in names:
            if name not in m:
                raise ValueError(f"{phase}: item {i} of the {which} has no {name}")
        out[i] = m
    return out


def merge(first_dir, again_dir, phases, expected=None) -> dict:
    """`expected` ({phase: ids}, the file the re-run was asked for) makes the merge refuse a re-run that is not exactly those items."""
    out = {}
    for phase in phases:
        first = _by_doc(first_dir, phase, "first run")
        again = _by_doc(again_dir, phase, "re-run")
        unknown = sorted(set(again) - set(first))
        if unknown:
            raise ValueError(f"{phase}: the re-run has items the first run does not: {unknown}")
        _check_same_question(phase, first, again)
        if expected is not None:
            _check_requested(phase, again, expected)
        merged = {i: (again[i] if i in again else s) for i, s in first.items()}
        names = sorted({k for s in first.values() for k in _metrics(s)})
        f_num, a_num = _numbers(phase, first, names, "first run"), _numbers(phase, again, names, "re-run")
        metrics, flips = {}, {}
        for name in names:
            before = [f_num[i][name] for i in first]
            after = [(a_num[i] if i in again else f_num[i])[name] for i in first]
            metrics[name] = {"before": round(sum(before) / len(before), 6), "after": round(sum(after) / len(after), 6)}
            gained = sorted(i for i in again if a_num[i][name] > f_num[i][name])
            lost = sorted(i for i in again if a_num[i][name] < f_num[i][name])
            flips[name] = {"gained": gained, "lost": lost}
        out[phase] = {"n": len(first), "rerun_n": len(again), "metrics": metrics, "flips": flips}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    it = sub.add_parser("items", help="list the items with an empty reply")
    it.add_argument("--run", required=True)
    it.add_argument("--phases", required=True)
    it.add_argument("--out", required=True)
    mg = sub.add_parser("merge", help="merge a re-run into the first run")
    mg.add_argument("--first", required=True)
    mg.add_argument("--again", required=True)
    mg.add_argument("--phases", required=True)
    mg.add_argument("--items", help="the file the re-run was asked for; the merge refuses a re-run that is not exactly those items")
    mg.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    phases = [p for p in a.phases.split(",") if p]
    if a.cmd == "items":
        items = cutoff_items(a.run, phases)
        for phase, ids in items.items():
            print(f"{phase}: {len(ids)} items")
        if not items:
            print("nothing to re-run: no empty replies")
            return 3
        Path(a.out).write_text(json.dumps(items) + "\n", encoding="utf-8")
        return 0
    expected = None
    if a.items:
        from orchard.bench_session import load_items
        expected = load_items(a.items)
    merged = merge(a.first, a.again, phases, expected=expected)
    Path(a.out).write_text(json.dumps(merged, indent=1) + "\n", encoding="utf-8")
    for phase, m in merged.items():
        for name, v in m["metrics"].items():
            print(f"{phase} {name}: {v['before']:.4f} -> {v['after']:.4f}  ({m['rerun_n']} of {m['n']} items re-run)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
