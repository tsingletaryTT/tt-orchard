#!/usr/bin/env python3
"""Stage 2's hardware test for a model of class `weights+sidecar`: the swap check, then the parity check.

The sidecar-parity skill copies this file next to `serve_and_compare.py` and `parity-run.sh` in the stage
directory, and writes `hw_test.json` with this one command, so the supervisor still runs a single command
under a single lease. It runs two children, one after the other, and always runs both:

1. `python3 serve_and_compare.py`: serves the swapped weights and compares the chip's tokens with the CPU
   reference. It stops its own server before it exits, so the board is free for the next child.
2. `bash parity-run.sh`: the launcher `prepare_parity.py` built. It runs hidden_parity.py in the bundle's
   python.

Each child runs in its own session (its own process group) and its output is streamed to this command's
output and kept in `evidence/swap-run.log` and `evidence/parity-run.log`. A `finally` stops each child's
whole group (SIGTERM, up to 60 s, then SIGKILL), and a SIGTERM sent to this command (the supervisor's
deadline) is turned into an exit, so the `finally` runs and no child outlives it.

Whatever happens, `evidence/sidecar-check.json` is written:

    {"swap_exit": 0, "parity_exit": 0, "seconds": {"swap": 612.3, "parity": 301.7},
     "result_draft": {<the swap draft: serves, server_ready_s, coherent, free_run_text, top1_agreement,
                       n_tokens, cache_dir>,
                      "sidecar_parity": <evidence/sidecar-parity.json, verbatim>,
                      "evidence": [<the swap draft's evidence paths>, "stages/2/evidence/sidecar-parity.json"]},
     "failure": "<what failed and the end of its output>"}          # only when something failed

The swap draft is `result_draft` of evidence/swap-check.json. Paths in `evidence` are relative to the run
directory, as the swap template writes them. If a half failed, its fields are missing from `result_draft`,
`failure` says why, and the exit code is the swap code if the swap failed, else the parity code, else 1.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

STAGE_DIR = Path(__file__).resolve().parent
EVIDENCE = STAGE_DIR / "evidence"
STOP_WAIT_S = 60.0
PIPE_GRACE_S = 2.0         # how long the output reader may keep going after the child has exited
TAIL_LINES = 25
SWAP_FIELDS = ("serves", "server_ready_s", "coherent", "free_run_text", "top1_agreement", "n_tokens", "cache_dir")


class Interrupted(BaseException):
    """SIGTERM or SIGINT reached this command. A BaseException, so no `except Exception` swallows it."""


def _on_signal(signum, frame):
    raise Interrupted(signal.Signals(signum).name)


def group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def stop_group(proc: subprocess.Popen) -> None:
    """Stop the child's whole process group. With start_new_session=True the child's pid is its group id."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + STOP_WAIT_S
    try:
        proc.wait(timeout=STOP_WAIT_S)
    except subprocess.TimeoutExpired:
        pass
    while time.monotonic() < deadline and group_alive(proc.pid):
        time.sleep(0.2)
    if group_alive(proc.pid):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


def run_child(argv: list[str], log_path: Path) -> tuple[int, str, float]:
    """Run one child in its own session, stream its output, return (exit code, tail of output, seconds)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    lines: list[str] = []
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(argv, cwd=STAGE_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                bufsize=1, start_new_session=True)

        def pump() -> None:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
                lines.append(line)
                del lines[:-TAIL_LINES]

        # A thread reads the output, so a process the child leaves behind (which keeps the pipe open) cannot
        # hold this command up once the child itself has exited.
        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        try:
            code = proc.wait()
            reader.join(timeout=PIPE_GRACE_S)          # the last lines the child printed
        finally:
            stop_group(proc)                  # also stops anything the child left behind
            reader.join(timeout=PIPE_GRACE_S)
    return code, "".join(lines).rstrip("\n"), time.monotonic() - started


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, ValueError) as exc:
        return None, f"{path.name} is not readable JSON: {exc}"


def run_dir_for() -> Path:
    for name in ("swap_config.json", "parity_config.json"):
        cfg, _ = read_json(STAGE_DIR / name)
        if isinstance(cfg, dict) and cfg.get("run_dir"):
            return Path(cfg["run_dir"])
    return STAGE_DIR.parent.parent          # stages/<n>/ -> the run directory


def rel(path: Path, run_dir: Path) -> str:
    try:
        return os.path.relpath(path, run_dir)
    except ValueError:
        return str(path)


def build_check(swap: dict | None, parity: dict | None) -> dict:
    """Assemble the sidecar-check.json content. Each half is {"exit": int | None, "tail": str, "seconds": float,
    "problem": str | None}; None means the half did not run."""
    run_dir = run_dir_for()
    draft: dict = {}
    failures: list[str] = []
    codes = {}
    seconds = {}
    evidence: list[str] = []
    for name, half in (("swap", swap), ("parity", parity)):
        codes[name] = half["exit"] if half else None
        seconds[name] = round(half["seconds"], 1) if half else 0.0
    # The swap half: result_draft of evidence/swap-check.json.
    swap_data, swap_err = read_json(EVIDENCE / "swap-check.json")
    if swap_data is not None:
        d = swap_data.get("result_draft") if isinstance(swap_data, dict) else None
        if isinstance(d, dict):
            draft.update({k: d[k] for k in SWAP_FIELDS if k in d})
            evidence += [e for e in d.get("evidence", []) if isinstance(e, str)]
        else:
            swap_err = "swap-check.json has no result_draft"
    # The parity half: evidence/sidecar-parity.json, verbatim.
    par_path = EVIDENCE / "sidecar-parity.json"
    par_data, par_err = read_json(par_path)
    if par_data is not None:
        draft["sidecar_parity"] = par_data
        evidence.append(rel(par_path, run_dir))
    if swap is None or swap["exit"] != 0:
        failures.append(f"the swap check (serve_and_compare.py) exited {swap['exit'] if swap else 'without running'}"
                        + (f": {swap['problem']}" if swap and swap.get("problem") else "")
                        + (f"\n{swap['tail']}" if swap and swap["tail"] else ""))
    elif swap_err:
        failures.append(f"the swap check exited 0 but {swap_err}")
    if parity is None or parity["exit"] != 0:
        failures.append(f"the parity check (parity-run.sh) exited {parity['exit'] if parity else 'without running'}"
                        + (f": {parity['problem']}" if parity and parity.get("problem") else "")
                        + (f"\n{parity['tail']}" if parity and parity["tail"] else ""))
    elif par_err:
        failures.append(f"the parity check exited 0 but {par_err}")
    draft["evidence"] = evidence
    out = {"swap_exit": codes["swap"], "parity_exit": codes["parity"], "seconds": seconds, "result_draft": draft}
    if failures:
        out["failure"] = "\n".join(failures)
    return out


def exit_code(check: dict) -> int:
    for key in ("swap_exit", "parity_exit"):
        if check[key] not in (0, None):
            return check[key]
    return 1 if "failure" in check else 0


def main() -> int:
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    swap = parity = None
    interrupted = None
    try:
        for name in ("serve_and_compare.py", "parity-run.sh"):
            if not (STAGE_DIR / name).is_file():
                half = {"exit": 127, "tail": "", "seconds": 0.0, "problem": f"{name} does not exist in {STAGE_DIR}"}
                if name == "serve_and_compare.py":
                    swap = half
                else:
                    parity = half
        if swap is None:
            code, tail, secs = run_child([sys.executable, str(STAGE_DIR / "serve_and_compare.py")],
                                         EVIDENCE / "swap-run.log")
            swap = {"exit": code, "tail": tail, "seconds": secs, "problem": None}
        if parity is None:
            code, tail, secs = run_child(["bash", str(STAGE_DIR / "parity-run.sh")], EVIDENCE / "parity-run.log")
            parity = {"exit": code, "tail": tail, "seconds": secs, "problem": None}
    except Interrupted as exc:
        interrupted = str(exc)
    finally:
        check = build_check(swap, parity)
        if interrupted:
            check["failure"] = f"interrupted by {interrupted}\n" + check.get("failure", "")
        EVIDENCE.mkdir(parents=True, exist_ok=True)
        (EVIDENCE / "sidecar-check.json").write_text(json.dumps(check, indent=2, ensure_ascii=False) + "\n",
                                                     encoding="utf-8")
    print(f"run_sidecar_checks: swap exit {check['swap_exit']}, parity exit {check['parity_exit']}"
          + ("; " + check["failure"].splitlines()[0] if "failure" in check else ""), flush=True)
    return 143 if interrupted else exit_code(check)


if __name__ == "__main__":
    sys.exit(main())
