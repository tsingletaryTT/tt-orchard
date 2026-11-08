#!/usr/bin/env python3
"""Serve the Berkelium CadQuery fine-tune on ONE board (two chips) with the container the 4-chip stage 2 check used.

Derived from notes/cad/berkelium-serve/serve-berkelium.py. Changes from the 4-chip command: devices 0 and 1 only (the first board),
MESH_DEVICE=P300, a separate tensor cache and kernel cache (the 4-chip ones hold other layouts), and serving limits that fit two chips
(max-model-len and max-num-seqs come from the command line). Run it under a gozer lease:
    gozer run --chips 2 --who "claude:can-cad" --reason "..." -- python3 serve-2chip.py --port 8061 --max-model-len 8192 --max-num-seqs 8
"""
import argparse, json, os, subprocess, sys
SRC = os.environ.get("BERKELIUM_SERVE_DIR") or sys.exit("set BERKELIUM_SERVE_DIR to the directory with docker-argv.json (the recorded 4-chip stage 2 command), model-dir/ and hf-isolated/")
HERE = os.environ.get("CAN_CAD_WORK", os.getcwd())      # the work directory: caches, base-clone/, staged/
ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8061)
ap.add_argument("--name", default="cad-berkelium-2chip")
ap.add_argument("--max-model-len", type=int, default=8192)
ap.add_argument("--max-num-seqs", type=int, default=8)
ap.add_argument("--devices", default=None, help="device indexes; default: the chips of the lease (TT_VISIBLE_DEVICES, mapped through `gozer status --json`)")
ap.add_argument("--mesh", default="P300")
ap.add_argument("--print", action="store_true")
a = ap.parse_args()
if a.devices is None:
    vis = [x for x in os.environ.get("TT_VISIBLE_DEVICES", "").split(",") if x]
    if not vis:
        sys.exit("no TT_VISIBLE_DEVICES: run under a gozer lease or pass --devices")
    st = json.loads(subprocess.check_output(["gozer", "status", "--json"]))
    idx = {c["bdf"]: c["dev_index"] for c in st["chips"]}
    a.devices = ",".join(str(idx[b]) for b in vis)
if 8000 <= a.port <= 8003 or 8100 <= a.port <= 8103 or a.port in (8011, 8051, 8071, 8081):
    sys.exit("pick another port")
argv = json.load(open(os.path.join(SRC, "docker-argv.json")))
cache = os.path.join(HERE, "tt_cache_2chip"); kcache = os.path.join(HERE, "kernel_cache_2chip")
out, i = [], 0
while i < len(argv):
    t = argv[i]
    nxt = argv[i + 1] if i + 1 < len(argv) else None
    if t == "--detach": i += 1; continue
    if t == "--name": out += ["--name", a.name]; i += 2; continue
    if t == "--label": i += 2; continue
    if t == "--publish": out += ["--publish", f"{a.port}:{a.port}"]; i += 2; continue
    if t == "--port": out += ["--port", str(a.port)]; i += 2; continue
    if t == "--device": i += 2; continue                       # re-added below for the chosen chips
    if t == "--env" and nxt.startswith("MESH_DEVICE="): out += ["--env", f"MESH_DEVICE={a.mesh}"]; i += 2; continue
    if t == "--volume" and nxt.endswith(":/tensor-cache"): out += ["--volume", f"{cache}:/tensor-cache"]; i += 2; continue
    if t == "--volume" and nxt.endswith(":/cache"): out += ["--volume", f"{kcache}:/cache"]; i += 2; continue
    if t == "--max-model-len": out += [t, str(a.max_model_len)]; i += 2; continue
    if t == "--max-num-seqs": out += [t, str(a.max_num_seqs)]; i += 2; continue
    out.append(t); i += 1
for d in a.devices.split(","):
    out[1:1] = []                                              # keep order simple: devices go right after `docker run`
out = out[:2] + ["--rm"] + sum([["--device", f"/dev/tenstorrent/{d}:/dev/tenstorrent/{d}"] for d in a.devices.split(",")], []) + out[2:]
if a.print:
    print(" ".join(json.dumps(x) if (" " in x or "{" in x) else x for x in out)); sys.exit(0)
sys.exit(subprocess.call(out))
