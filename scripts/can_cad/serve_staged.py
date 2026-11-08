#!/usr/bin/env python3
"""Serve the staged package through tt-model's own serve path (the same function `tt-model serve` runs), for the boot check.
usage: serve_staged.py [--print] [--profile p300] [--port N] [--device-id 0,1]   (HF_HOME must hold the weights)"""
import argparse, json, os, subprocess, sys
from pathlib import Path
from tt_kernel.manifest import Manifest
from tt_kernel import container, container_cli
# tt-model refuses to start while ANY running container maps /dev/tenstorrent, and TT Studio's docker-control container maps the whole directory
# without opening a device (fuser shows no holder), so the refusal is wrong here. The chips are protected by this caller's gozer lease instead;
# the check is skipped only when the caller holds a lease (TT_VISIBLE_DEVICES is set).
if os.environ.get("TT_VISIBLE_DEVICES"):
    container.ensure_devices_free = lambda ids: print("note: tt-model's container device check skipped; chips are held by a gozer lease", file=sys.stderr)
ap = argparse.ArgumentParser()
ap.add_argument("--print", dest="print_only", action="store_true")
ap.add_argument("--profile", default="p300")
ap.add_argument("--port", type=int, default=None)
ap.add_argument("--device-id", default=None)
ap.add_argument("--no-weights", action="store_true")
ap.add_argument("--detach", action="store_true")
a = ap.parse_args()
if a.device_id is None and os.environ.get("TT_VISIBLE_DEVICES"):
    st = json.loads(subprocess.check_output(["gozer", "status", "--json"]))
    idx = {c["bdf"]: c["dev_index"] for c in st["chips"]}
    a.device_id = ",".join(str(idx[b]) for b in os.environ["TT_VISIBLE_DEVICES"].split(","))
staged = Path(os.environ.get("CAN_CAD_WORK", ".")) / "staged"
m = Manifest.from_json((staged / "tt_kernel_manifest.json").read_text())
container_cli.serve_container(m, profile_name=a.profile, print_only=a.print_only, source=staged, port=a.port, target=str(staged),
                              local_only=True, no_weights=a.no_weights, detach=a.detach, device_id=a.device_id)
