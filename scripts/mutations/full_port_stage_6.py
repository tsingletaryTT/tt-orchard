#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""One mutation per guard of stage 6 of a full port (issue 329). Each one edits a source file, runs the named tests, and expects a failure.

    python3 scripts/mutations/full_port_stage_6.py [name ...]

It refuses to start on a dirty working tree (commit first), restores every file in a `finally` and on SIGTERM, and checks that the baseline test
run is green before it reads any row. A row is `RED` when a test failed under the mutation (the guard has a test) and `SURVIVED` when none did.
Run it under the resource wrapper the project uses (systemd-run with a memory cap, nice, timeout).
"""
import os
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
Q = "orchard/full_port_quality.py"
S = "orchard/supervisor.py"
QS = "orchard/full_port_quality_stage.py"
T = "orchard/skills/full-port-templates/port_quality.py"
UNIT, SUP = "tests/test_full_port_quality.py", "tests/test_supervisor_full_port_quality.py"

# name: (file, old text, new text, tests to run)
MUTATIONS = {
    "gap-bar": (Q, "    if ppl[\"top5_gap_diff\"] > GAP_MAX:", "    if False:", UNIT),
    "gap-bar-waived-by-rehearsal": (Q, "    if ppl[\"top5_gap_diff\"] > GAP_MAX:", "    if ppl[\"top5_gap_diff\"] > GAP_MAX and not rehearsal:", UNIT),
    "top1-bar": (Q, "        if agree < TOP1_MIN:", "        if False:", UNIT),
    "coherent-needed": (Q, "            elif want is not True:\n                reasons.append(f\"coherent must be true", "            elif False:\n                reasons.append(f\"coherent must be true", UNIT),
    "derive-readable-floor": (Q, "    if not result[\"coherent\"] and not rehearsal:", "    if False:", UNIT),
    "repeat-rule": (Q, "    if bench[\"repeats\"] < resident.MIN_REPEATS:", "    if False:", UNIT),
    "warmup-rule": (Q, "    if bench[\"warmup_runs\"] < resident.WARMUP_RUNS:", "    if False:", UNIT),
    "share-bound": (Q, "    if not _finite(share) or not 0 < share <= SHARE_MAX:", "    if False:", UNIT),
    "speed-is-steps-over-time": (Q, "        elif not math.isclose(r[\"steps\"] / r[\"decode_s\"], r[\"decode_tok_s\"], rel_tol=1e-6):", "        elif False:", UNIT),
    "result-recomputed": (Q, "        elif not _same(result.get(key), want):", "        elif False:", UNIT),
    "top5-file-match": (Q, "    _top5_reasons(record, run_dir, reasons)\n    arrays", "    arrays", UNIT),
    "template-hash": (Q, "    elif mine != shipped:", "    elif False:", UNIT),
    "output-mark": (Q, "    if not marks or marks[-1] != full_port._sha(stage_dir / RECORD_FILE):", "    if False:", UNIT),
    "command-fixed": (Q, "        if marker.get(\"command\") != command():", "        if False:", UNIT),
    "chip-count": (Q, "        elif expected_chips is not None and (len(rc) != expected_chips or len(mcs) != expected_chips):", "        elif False:", UNIT),
    "seeded-hashes": (Q, "    _seed_reasons(record, run_dir, seeded, source_stage, reasons)", "    pass", UNIT),
    "source-record": (Q, "    if not theirs or theirs != mine:", "    if False:", UNIT),
    "prefill-resets": (Q, "    elif free[0] != chip[0]:", "    elif False:", UNIT),
    "test-passed": (Q, "        if not (type(code) is int and code == 0 and timed_out is False):", "        if False:", UNIT),
    "template-benchmark-repeats": (T, "REPEATS = 3 ", "REPEATS = 2 ", UNIT),
    "template-forged-mark": (T, "            if MARK.encode() in raw:\n                FORGED.set()", "            if False:\n                FORGED.set()", UNIT),
    "template-links": (T, "            if os.path.islink(os.path.join(root, name)):", "            if False:", UNIT),
    "supervisor-gate-hook": (S, "            return full_port_quality_stage.gate(self, spec) ", "            return spec.gate ", SUP),
    "supervisor-no-seeded-in-gate": (QS, "\"seeded\": seeded(sup), ", "\"seeded\": [], ", SUP),
    "supervisor-rehearsal-from-ledger": (QS, "    if full_port_rehearsal(sup.ledger.read()):\n        kwargs", "    if False:\n        kwargs", SUP),
    "supervisor-marker-from-ledger": (S, "\"full-port-mesh\", \"full-port-quality\"):\n            # One hardware test, one marker.", "\"full-port-mesh\"):\n            # One hardware test, one marker.", SUP),
    "supervisor-source-stage": (QS, "    return 3 if sup._stage_3_chips() else 2", "    return 2", SUP),
}


def run_tests(tests: str) -> bool:
    """True when the tests pass."""
    p = subprocess.run([sys.executable, "-m", "pytest", *tests.split(), "-q", "-x", "-p", "no:cacheprovider"], cwd=ROOT, capture_output=True, text=True)
    return p.returncode == 0


def main(argv) -> int:
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if dirty:
        print("refusing to start: the working tree is not clean. Commit first, so the restore has something to restore to.\n" + dirty)
        return 2
    names = argv or list(MUTATIONS)
    unknown = [n for n in names if n not in MUTATIONS]
    if unknown:
        print("unknown mutations:", unknown)
        return 2
    originals: dict[Path, bytes] = {}

    def restore(*_):
        for path, data in originals.items():
            path.write_bytes(data)
        subprocess.run(["git", "checkout", "--", "."], cwd=ROOT)
    signal.signal(signal.SIGTERM, lambda *a: (restore(), os._exit(143)))
    try:
        if not run_tests(f"{UNIT} {SUP}"):
            print("the baseline is not green; no mutation row means anything until it is")
            return 2
        survived = 0
        for name in names:
            rel, old, new, tests = MUTATIONS[name]
            path = ROOT / rel
            text = path.read_text()
            if text.count(old) != 1:
                print(f"{name}: SKIPPED (the text to mutate occurs {text.count(old)} times)")
                survived += 1
                continue
            originals[path] = text.encode()
            path.write_text(text.replace(old, new))
            green = run_tests(tests)
            path.write_bytes(originals.pop(path))
            print(f"{name}: {'SURVIVED' if green else 'RED'}")
            survived += green
        return 1 if survived else 0
    finally:
        restore()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
