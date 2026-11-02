# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Run the scoring check by hand, the way the supervisor would, and put its result through the gate (issue 375).

The stage machine does not run this check yet (issue 378), so there is nobody to build the stage directory, run the template under a lease
and write `test-result.json`. This script does those three steps for a run on a board that an operator or an agent already leased:

    python3 scripts/scoring_check/run_gate.py setup --run-dir DIR --reference-dir REF --snapshot SNAP
    python3 scripts/scoring_check/run_gate.py run   --run-dir DIR --chips BDF[,BDF] --server-url http://127.0.0.1:PORT

`setup` copies the stage 1 files (scoring-pairs.json and scoring-reference.json, from `orchard.scoring_reference`) into DIR/stages/1/
evidence/reference/, the shipped template and this directory's port_model.py into DIR/stages/2/, and writes the config with the default
bounds. `run` starts the supervisor's command in DIR with TT_VISIBLE_DEVICES set to --chips, writes the output file and the marker the
gate reads, and runs `orchard.scoring_check.check`. It prints the template's output, the reasons the gate gives and the exit code is 0
only when the test exited 0 and the gate gave no reason.

What this does not show: --chips is a claim that the caller holds a lease on those chips. This script does not take one. The chips named
there are the ones the template opens, so with the HTTP port the model server must hold a different chip of the leased board.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from orchard import full_port, scoring_check          # noqa: E402  (after the path line)

HERE = Path(__file__).resolve().parent
FILES = ("scoring-pairs.json", "scoring-reference.json")


def setup(run_dir: Path, reference_dir: Path, snapshot: Path, port_source: Path = HERE / "port_model.py") -> Path:
    """Build DIR/stages/1 and DIR/stages/2. Returns the stage 2 directory. An existing run directory is refused."""
    if run_dir.exists():
        raise SystemExit(f"{run_dir} exists: use a new directory for each run")
    ref = run_dir / "stages" / "1" / "evidence" / "reference"
    stage = run_dir / "stages" / "2"
    ref.mkdir(parents=True)
    (stage / "port").mkdir(parents=True)
    for name in FILES:
        shutil.copyfile(reference_dir / name, ref / name)
    shutil.copyfile(port_source, stage / "port" / "port_model.py")
    shutil.copyfile(scoring_check.TEMPLATE, stage / scoring_check.TEMPLATE_NAME)
    (stage / scoring_check.CONFIG_NAME).write_text(json.dumps(scoring_check.draft_config(str(snapshot)), indent=2) + "\n")
    return stage


def run(run_dir: Path, chips: list, server_url: str, timeout_s: float = scoring_check.DEADLINE_S, env: dict | None = None):
    """Run the supervisor's command in `run_dir`, write the output file and `test-result.json`, and apply the gate.
    Returns (completed process, reasons)."""
    stage = run_dir / "stages" / "2"
    cmd = scoring_check.COMMAND.split()
    cmd[0] = sys.executable
    full_env = dict(os.environ if env is None else env)
    full_env.pop("HF_TOKEN", None)                          # the check needs no credential
    full_env.update({"TT_VISIBLE_DEVICES": ",".join(chips), "ORCHARD_SCORE_SERVER_URL": server_url})
    t0 = time.monotonic()
    timed_out = False
    try:
        done = subprocess.run(cmd, cwd=run_dir, env=full_env, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        done = subprocess.CompletedProcess(cmd, None, (exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
                                           (exc.stderr or b"").decode() if isinstance(exc.stderr, bytes) else (exc.stderr or ""))
    out = stage / "evidence" / "hw-test-output.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(done.stdout + done.stderr)
    marker = {"command": scoring_check.COMMAND, "returncode": done.returncode, "timed_out": timed_out,
              "seconds": round(time.monotonic() - t0, 1), "chips": list(chips), "output": full_port.file_record(run_dir, out)}
    (stage / "test-result.json").write_text(json.dumps(marker))
    reasons, _ = scoring_check.check(stage, run_dir)
    return done, reasons


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup")
    s.add_argument("--run-dir", required=True, type=Path)
    s.add_argument("--reference-dir", required=True, type=Path)
    s.add_argument("--snapshot", required=True, type=Path)
    r = sub.add_parser("run")
    r.add_argument("--run-dir", required=True, type=Path)
    r.add_argument("--chips", required=True, help="the chips the template opens, comma separated (bus addresses, as gozer prints them)")
    r.add_argument("--server-url", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "setup":
        print(setup(args.run_dir, args.reference_dir, args.snapshot))
        return 0
    done, reasons = run(args.run_dir, [c for c in args.chips.split(",") if c], args.server_url)
    print(done.stdout + done.stderr)
    print("exit code of the test:", done.returncode)
    print("gate reasons:", json.dumps(reasons, indent=2) if reasons else "none (the gate passes)")
    return 0 if done.returncode == 0 and not reasons else 1


if __name__ == "__main__":
    sys.exit(main())
