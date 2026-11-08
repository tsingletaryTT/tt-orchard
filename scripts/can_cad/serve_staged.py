#!/usr/bin/env python3
"""Serve the staged package through tt-model's own serve path (the function `tt-model serve` runs), for the boot check.
usage: serve_staged.py [--print] [--profile p300] [--port N] [--device-id 0,1]   (HF_HOME must hold the weights; CAN_CAD_WORK holds staged/)

tt-model refuses to start while any running container or host process counts as holding a chip, and on this machine the refusal came up with
no holder (issue 560). When this caller holds a live gozer lease, the check is replaced by gozer's own: every requested device must belong to
that lease and gozer must still show the chips held by the lease's owner pid (`lease_covers`). Without that proof the check stays on."""
import argparse
import json
import os
import subprocess
import sys

LIVE_STATES = ("HELD", "CLAIMED", "HELD-FOREIGN")


def lease_covers(requested: list, env: dict, status: dict) -> tuple:
    """(True, "") when every requested device index belongs to the live lease named by the environment the lease wrapper exported
    (TT_VISIBLE_DEVICES, TT_LEASE_OWNER_PID, TT_LEASE_DEV_INDICES) and `gozer status --json` shows each of those chips held by that owner pid;
    otherwise (False, why)."""
    visible = [b for b in env.get("TT_VISIBLE_DEVICES", "").split(",") if b]
    owner = env.get("TT_LEASE_OWNER_PID", "")
    leased = [int(x) for x in env.get("TT_LEASE_DEV_INDICES", "").split(",") if x.strip().isdigit()]
    if not visible or not owner.isdigit() or not leased:
        return False, "no gozer lease in the environment (run under lease_run.sh)"
    outside = sorted(set(requested) - set(leased))
    if outside:
        return False, f"device(s) {outside} are not in the lease ({leased})"
    by_bdf = {c.get("bdf"): c for c in status.get("chips", []) if isinstance(c, dict)}
    for bdf in visible:
        c = by_bdf.get(bdf)
        if c is None or c.get("state") not in LIVE_STATES or c.get("pid") != int(owner):
            return False, f"chip {bdf} is not held by lease owner pid {owner} (gozer shows {None if c is None else (c.get('state'), c.get('pid'))})"
    return True, ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", dest="print_only", action="store_true")
    ap.add_argument("--profile", default="p300")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--device-id", default=None)
    ap.add_argument("--no-weights", action="store_true")
    ap.add_argument("--detach", action="store_true")
    a = ap.parse_args(argv)
    from pathlib import Path
    from tt_kernel import container, container_cli
    from tt_kernel.manifest import Manifest
    status = {}
    if os.environ.get("TT_VISIBLE_DEVICES"):
        status = json.loads(subprocess.check_output(["gozer", "status", "--json"]))
        if a.device_id is None:
            a.device_id = os.environ.get("TT_LEASE_DEV_INDICES") or None
    if a.device_id is not None and os.environ.get("TT_VISIBLE_DEVICES"):
        ok, why = lease_covers([int(x) for x in a.device_id.split(",")], os.environ, status)
        if ok:
            container.ensure_devices_free = lambda ids: print("note: tt-model's container device check skipped; the chips are held by this gozer lease", file=sys.stderr)
        else:
            print(f"note: tt-model's container device check stays on: {why}", file=sys.stderr)
    staged = Path(os.environ.get("CAN_CAD_WORK", ".")) / "staged"
    m = Manifest.from_json((staged / "tt_kernel_manifest.json").read_text())
    container_cli.serve_container(m, profile_name=a.profile, print_only=a.print_only, source=staged, port=a.port, target=str(staged),
                                  local_only=True, no_weights=a.no_weights, detach=a.detach, device_id=a.device_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
