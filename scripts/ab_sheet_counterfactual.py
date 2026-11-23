#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Which finished kernel-probe attempts used a form the old cheat sheet called "the hardware form", and what happens to them if
that form is replaced by the form the chip compilers and the simulator both take.

    python3 scripts/ab_sheet_counterfactual.py PROBE_DIR [PROBE_DIR ...] [--out FILE] [--no-compile]

PROBE_DIR is the output of one `tt-orchard kernel-probe` run, or a directory of such runs. Per attempt the table gives the judged
kernel's forms (counted in the source text), the simulator verdict the probe recorded and, for a kernel that uses a wrong-claim form, a
mechanical rewrite and the verdict of the simulator's unedited check on the rewrite. The rewrite changes only the namespace of
`fill` and `broadcast` (`ttl.math.` to `ttl.block.`) and adds `shape=(1, 1)` to a `reduce_sum` or `reduce_max` call that has none (a
reduction over the last dimension of a row of tiles leaves one tile). It does not repair anything else, so a rewrite that still fails
shows what else was wrong. The rewrite says what happens to one kernel. It does not say what the coder would have written.

Everything runs on the CPU: the simulator, and (unless --no-compile) the chip packages' own compilers with no device
(orchard/ttlang_hwcompile.py). No model is called.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from orchard import kernelprobe, safeio, ttlang_hwcompile, ttlang_pair  # noqa: E402

REPORT = "kernel-probe-report.json"
SCHEMA = "orchard-ab-sheet-counterfactual/1"

# The forms the old sheet got wrong, and the forms that replace them. The pattern for a reduction call without `shape=` stops at the
# first `)`; the calls in the probe's kernels have no nested parentheses in their arguments.
FORMS = {
    "ttl.math.fill": re.compile(r"ttl\.math\.fill\b"),
    "ttl.math.broadcast": re.compile(r"ttl\.math\.broadcast\b"),
    "reduce without shape": re.compile(r"ttl\.math\.reduce_(?:sum|max)\((?![^)]*\bshape\s*=)[^)]*\)"),
    "ttl.block.fill": re.compile(r"ttl\.block\.fill\b"),
    "ttl.block.broadcast": re.compile(r"ttl\.block\.broadcast\b"),
    "reduce with shape": re.compile(r"ttl\.math\.reduce_(?:sum|max)\([^)]*\bshape\s*="),
}
WRONG = ("ttl.math.fill", "ttl.math.broadcast", "reduce without shape")


def count_forms(text: str) -> dict:
    return {name: len(pattern.findall(text)) for name, pattern in FORMS.items()}


def rewrite(text: str) -> str:
    """The mechanical rewrite (see the module text). Idempotent: a second pass changes nothing."""
    text = FORMS["ttl.math.fill"].sub("ttl.block.fill", text)
    text = FORMS["ttl.math.broadcast"].sub("ttl.block.broadcast", text)
    return FORMS["reduce without shape"].sub(lambda m: m.group(0)[:-1].rstrip().rstrip(",") + ", shape=(1, 1))", text)


def probe_dirs(paths) -> list[Path]:
    found = []
    for path in map(Path, paths):
        if (path / REPORT).is_file():
            found.append(path)
        elif path.is_dir():
            found += sorted(p for p in path.iterdir() if (p / REPORT).is_file())
    return found


def simulator_check(attempt_dir: Path, op: str, kernel_text: str, work: Path) -> dict:
    """The probe's own unedited-copy check, on a copy of the attempt directory whose kernel is `kernel_text`."""
    copy = work / "attempt"
    shutil.copytree(attempt_dir, copy, ignore=shutil.ignore_patterns("__pycache__", "home", "log"))
    (copy / "kernel.py").write_text(kernel_text, encoding="utf-8")
    args = SimpleNamespace(op=op, library="none", src_dir=None, venv_dir=None, tt_lang_skill_dir=None, reference_source="checkout")
    env = kernelprobe.default_env(args)
    check = kernelprobe.authoritative_check(copy, op=op, env=env, tt_lang=kernelprobe.tt_lang_facts(env))
    return {"passed": check.get("passed") is True, "category": check.get("category"), "first_line": str(check.get("first_line"))[:300]}


def judged_text(probe_dir: Path, attempt: dict) -> str | None:
    op, n = attempt.get("op"), attempt.get("attempt")
    want = (attempt.get("provenance") or {}).get("kernel_sha256")
    path = kernelprobe.verdict_dir(probe_dir, op, n) / kernelprobe.CLEAN_DIR / "kernel.py"
    try:
        data = safeio.read_bytes(path, max_bytes=ttlang_hwcompile.MAX_KERNEL_BYTES)
    except safeio.SafeReadError:
        return None
    return data.decode("utf-8") if hashlib.sha256(data).hexdigest() == want else None


def check_run(probe_dir: Path, targets, *, compile_it: bool) -> list[dict]:
    report = safeio.read_json(probe_dir / REPORT, max_bytes=kernelprobe.RECORD_MAX_BYTES, expect=dict)
    rows = []
    for attempt in (a for a in report.get("attempts", []) if isinstance(a, dict)):
        op, n = attempt.get("op"), attempt.get("attempt")
        text = judged_text(probe_dir, attempt)
        row = {"run": probe_dir.name, "op": op, "attempt": n, "passed_sim": attempt.get("passed_sim") is True,
               "first_line": str(attempt.get("first_line") or attempt.get("note") or "")[:200]}
        if text is None:
            row["problem"] = "the judged kernel is missing or its bytes do not match the attempt's record"
            rows.append(row)
            continue
        row["forms"] = count_forms(text)
        row["uses_wrong_claim"] = any(row["forms"][name] for name in WRONG)
        if row["uses_wrong_claim"]:
            new = rewrite(text)
            row["rewrite_changed"] = new != text
            with tempfile.TemporaryDirectory(prefix="ab-cf-") as tmp:
                row["rewrite_simulator"] = simulator_check(probe_dir / op / f"attempt-{n}", op, new, Path(tmp))
                if compile_it and row["rewrite_simulator"]["passed"]:
                    kernel = Path(tmp) / "rewrite.py"
                    kernel.write_text(new, encoding="utf-8")
                    spec = kernelprobe.load_spec(op)
                    row["rewrite_compile"] = {}
                    for python, identity in targets:
                        got = ttlang_hwcompile.compile_check(python, kernel, ttlang_hwcompile.spec_shapes(spec), identity=identity)
                        row["rewrite_compile"][identity["version"]] = {"category": got["category"], "first_line": got["first_line"]}
        rows.append(row)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("probe_dir", nargs="+")
    ap.add_argument("--out")
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args(argv)
    targets = []
    if not args.no_compile:
        for python in (ttlang_pair.default_venv() / "bin" / "python", Path.home() / ".tenstorrent-venv" / "bin" / "python"):
            try:
                targets.append((python, ttlang_hwcompile.package_identity(python)))
            except ttlang_hwcompile.HwCompileError as exc:
                print(f"skipped {python}: {exc}", file=sys.stderr)
    rows = [row for d in probe_dirs(args.probe_dir) for row in check_run(d, targets, compile_it=not args.no_compile)]
    print("| run | simulator (recorded) | wrong-claim forms | rewrite: simulator | rewrite: chip compilers |")
    print("|---|---|---|---|---|")
    for r in rows:
        wrong = ", ".join(f"{k} x{r['forms'][k]}" for k in WRONG if r.get("forms", {}).get(k)) or "none"
        sim = r.get("rewrite_simulator")
        sim_text = "-" if sim is None else ("pass" if sim["passed"] else f"fail: {sim['first_line'][:80]}")
        comp = "-" if "rewrite_compile" not in r else "; ".join(f"{v}: {c['category']}" + (f" ({c['first_line'][:70]})" if c["category"] != "compiled" else "") for v, c in r["rewrite_compile"].items())
        print(f"| {r['run']} | {'pass' if r['passed_sim'] else 'fail'} | {wrong} | {sim_text} | {comp} |")
    if args.out:
        safeio.atomic_write(Path(args.out), json.dumps({"schema": SCHEMA, "rows": rows}, indent=1) + "\n", mode=0o644)
    return 0


if __name__ == "__main__":
    sys.exit(main())
