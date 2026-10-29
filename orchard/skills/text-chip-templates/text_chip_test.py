# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The one command of a text stage's hardware phase: run the commands of `text_chip_config.json` in order under the lease the supervisor holds.

    python text_chip_test.py --config STAGE_DIR/text_chip_config.json

Each command is an argv list from the runtime adapter (orchard/text_runtime.py). A command opens and closes the device itself, one process at a time, so the
chip is free between commands. The orchestrator stops at the first command that fails, because the commands after it would spend the lease on a run that the
gate refuses. Each command's output goes to `log/<name>.log` in the stage directory, and `text_chip_run.json` records each command's exit status, seconds and the
files it was to write. It opens no device and imports nothing but the standard library. The exit status is 0 when every command ended with 0 and left its files.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    config = Path(a.config)
    stage = config.parent
    cfg = json.loads(config.read_text(encoding="utf-8"))
    (stage / "log").mkdir(exist_ok=True)
    done = []
    status = 0
    for cmd in cfg["commands"]:
        t0 = time.time()
        log = stage / "log" / f"{cmd['name']}.log"
        with open(log, "wb") as out:
            code = subprocess.run(cmd["argv"], stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, env=dict(os.environ), cwd=stage).returncode
        left = [f for f in cmd.get("outputs", []) if (stage / f).is_file()]
        missing = [f for f in cmd.get("outputs", []) if f not in left]
        done.append({"name": cmd["name"], "exit": code, "seconds": round(time.time() - t0, 1), "missing_outputs": missing})
        print(f"{cmd['name']}: exit {code} in {done[-1]['seconds']} s" + (f"; missing {missing}" if missing else ""), flush=True)
        if code != 0 or missing:
            status = 1
            break
    (stage / "text_chip_run.json").write_text(json.dumps({"commands": done, "ok": status == 0}, indent=1) + "\n", encoding="utf-8")
    return status


if __name__ == "__main__":
    sys.exit(main())
