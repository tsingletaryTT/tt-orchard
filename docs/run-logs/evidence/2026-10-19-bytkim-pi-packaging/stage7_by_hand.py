#!/usr/bin/env python3
"""Stage 7 by hand for a finished weights-only run, the way the supervisor does it (orchard/supervisor.py, _package_body).

    python3 stage7_by_hand.py stage                      # stage and scrub every profile, install a copy of the required one
    gozer run --chips 2 --who ... --reason ... -- timeout 4200 python3 stage7_by_hand.py boot     # the boot check, on a leased board
    python3 stage7_by_hand.py finish                     # record the result, write the cards and the publish text, run the gate

The boot check gets what the supervisor gives a hardware test: the agent shells' environment (no tokens, HOME inside the run directory), the
leased chips in TT_VISIBLE_DEVICES, and the shared kernel cache. Nothing here uploads, publishes or pushes. Not recorded in the run's ledger.
"""
import json
import os
import socket
import sys
import time
from pathlib import Path

from orchard import package
from orchard.agent import agent_env, spawn_checked
from orchard.package import PackageError
from orchard.stages import evidence_record, gate_package

RUN = Path(os.environ.get("ORCHARD_PACKAGE_RUN", Path.home() / "orchard-runs" / "bytkim--qwen3.8-27b-pi"))
STAGE = RUN / "stages" / "7"
MODELS = Path.home() / ".cache" / "tt-model" / "models"
NAMESPACE = os.environ.get("ORCHARD_PACKAGE_NAMESPACE", "episod")
KERNEL_CACHE = Path.home() / "orchard-runs" / "cache" / "kernels"


def stage() -> int:
    staged = package.stage_all(RUN, STAGE, namespace=NAMESPACE, models_root=MODELS, env=agent_env(RUN), hostname=socket.gethostname())
    print("staged:", [(p["name"], p["required"], p["mesh"]) for p in staged["profiles"]], "skipped:", staged["skipped_profiles"])
    return 0


def boot() -> int:
    test = json.loads((STAGE / "hw_test.json").read_text())
    chips = os.environ["TT_VISIBLE_DEVICES"]                      # set by gozer run for the leased chips
    env = agent_env(RUN)
    env["TT_VISIBLE_DEVICES"] = chips
    env.pop("TT_METAL_VISIBLE_DEVICES", None)
    env["TT_METAL_CACHE"] = str(KERNEL_CACHE)
    out_path = STAGE / "evidence" / "hw-test-output.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(out_path, "wb") as out:
        code, timed_out = spawn_checked(test["command"], RUN, env, float(test["deadline_s"]), out)
    result = {"command": test["command"], "returncode": code, "timed_out": timed_out, "seconds": round(time.time() - t0, 3),
              "chips": chips.split(","), "output": evidence_record(RUN, out_path)}
    tmp = STAGE / "test-result.json.tmp"
    tmp.write_text(json.dumps(result, indent=2))
    os.replace(tmp, STAGE / "test-result.json")
    print("boot check: returncode", code, "timed_out", timed_out, "seconds", result["seconds"])
    return 0 if code == 0 else 1


def finish() -> int:
    package.finish(RUN, STAGE, hostname=socket.gethostname())
    gate = gate_package(STAGE, RUN)
    print("gate_package ok:", gate.ok)
    for r in gate.reasons:
        print("  -", r)
    return 0 if gate.ok else 1


if __name__ == "__main__":
    try:
        sys.exit({"stage": stage, "boot": boot, "finish": finish}[sys.argv[1]]())
    except (KeyError, IndexError):
        sys.exit("usage: stage7_by_hand.py stage|boot|finish")
    except PackageError as exc:
        sys.exit(f"packaging: {exc}")
