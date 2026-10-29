# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Checks the machine before the first start: `bash run.sh --self-test [--open-device]`.

One line per check, `ok`, `warn` or `FAIL`, then a summary line. Nothing here opens the device unless `--open-device` is given (then the
device is opened and closed once, which needs a free chip and, on a shared machine, a lease). Exit status 1 when a check failed."""
import argparse
import importlib.util
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

MODEL = "ibm-granite/granite-4.0-h-1b"
REVISION = "d18cca4c121edb87d022116d281ce212c9136f57"
NEED_GB = 8        # weights (about 3 GB) with room for the cache's temporary files
PACKAGES = ("torch", "numpy", "safetensors", "tokenizers", "jinja2", "huggingface_hub", "loguru", "networkx", "yaml", "pandas")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--open-device", action="store_true")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 20110)))
    a = ap.parse_args()
    out = []

    def line(status, name, detail):
        out.append(status)
        print(f"{status:4}  {name}: {detail}", flush=True)

    v = sys.version_info
    line("ok" if v >= (3, 10) else "FAIL", "python", f"{v.major}.{v.minor}.{v.micro} (tested with 3.12)")
    missing = [p for p in PACKAGES if importlib.util.find_spec(p) is None]
    line("FAIL" if missing else "ok", "python packages", "missing: " + ", ".join(missing) + " (pip install -r requirements.txt)" if missing else "all present")
    r = subprocess.run([sys.executable, "-c", "import ttnn"], capture_output=True, text=True, timeout=120)
    if r.returncode == 0:
        line("ok", "import ttnn", "works (this does not open the device)")
    else:
        tail = (r.stderr.strip().splitlines() or ["no message"])[-1][:160]
        line("FAIL", "import ttnn", f"{tail}; set TT_METAL_HOME to a tt-metal checkout with a built ttnn (run.sh adds it to PYTHONPATH)")
    devs = sorted(Path("/dev/tenstorrent").glob("*")) if Path("/dev/tenstorrent").is_dir() else []
    line("ok" if devs else "FAIL", "chip visibility", f"{len(devs)} device node(s) in /dev/tenstorrent" + (f"; TT_VISIBLE_DEVICES={os.environ['TT_VISIBLE_DEVICES']}" if os.environ.get("TT_VISIBLE_DEVICES") else ""))
    hf = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    snap = hf / "hub" / ("models--" + MODEL.replace("/", "--")) / "snapshots" / REVISION
    have = (snap / "model.safetensors").exists()
    line("ok" if have else "warn", "pinned weights", f"revision {REVISION[:8]} {'is in the cache' if have else 'is not in the cache yet; the first start downloads about 3 GB into ' + str(hf)}")
    probe = hf if hf.exists() else hf.parent if hf.parent.exists() else Path("/")
    free = shutil.disk_usage(probe).free / 1e9
    line("ok" if have or free >= NEED_GB else "FAIL", "free disk", f"{free:.0f} GB free where the cache lives" + ("" if have or free >= NEED_GB else f" (need about {NEED_GB} GB for the download)"))
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", a.port))
        line("ok", "port", f"127.0.0.1:{a.port} is free")
    except OSError as e:
        line("FAIL", "port", f"127.0.0.1:{a.port} is not available ({e.strerror}); set PORT")
    finally:
        s.close()
    tm = os.environ.get("TT_METAL_HOME")
    if tm and (Path(tm) / ".git").exists():
        d = subprocess.run(["git", "-C", tm, "describe", "--tags", "--always"], capture_output=True, text=True).stdout.strip()
        line("ok", "tt-metal", f"{d} (tested with v0.80.0)")
    if a.open_device:
        r = subprocess.run([sys.executable, "-c", "import ttnn; d = ttnn.open_device(device_id=0); ttnn.close_device(d)"], capture_output=True, text=True, timeout=300)
        line("ok" if r.returncode == 0 else "FAIL", "open device", "opened and closed device 0" if r.returncode == 0 else (r.stderr.strip().splitlines() or ["failed"])[-1][:160])
    else:
        line("ok", "open device", "skipped (add --open-device to test it; needs a free chip)")
    bad = out.count("FAIL")
    print(f"self-test: {bad} failed, {out.count('warn')} warning(s)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
