"""Run only the hardware phase of a finished kernel-probe directory (one smoke test, then each simulator pass), with the
interpreter that `kernel-probe` itself would choose. Used when the board was busy at the end of the probe's own hardware phase.
usage: ORCHARD_PROBE_CHIP=2 python3 hw_only.py PROBE_DIR OP PROBE_ID"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from orchard import kernelprobe as kp, ttlang_pair

probe_dir, op, probe_id = sys.argv[1:4]
choice = ttlang_pair.resolve_hardware_python(None, fallback=kp.default_shared_python())
print("interpreter source:", choice.source, choice.warning or "")
import time
env = kp.default_env(argparse.Namespace())
for tries in range(1, 13):                       # the board is shared: when gozer queues the request, try again in 5 minutes
    attempts, _ = kp.load_attempts(probe_dir, ops=[op], probe_id=probe_id)
    state = {}
    try:
        res = kp.run_hardware(attempts, probe_dir=probe_dir, env=env, shared_python=choice.python, state=state)
        if not res.get("queued"):
            break
        print("try", tries, "smoke test queued:", res["queued"], flush=True)
        time.sleep(300)
        continue
    except kp.ProbeBlocked as exc:
        res = {"blocked": str(exc), "smoke": state.get("smoke")}
        print("try", tries, "blocked:", exc, flush=True)
        if "queued" not in str(exc):
            break
        time.sleep(300)
print(json.dumps(res, indent=1)[:1500])
for a in attempts:
    print(a["attempt"], a.get("passed_sim"), json.dumps(a.get("hw"))[:700])
