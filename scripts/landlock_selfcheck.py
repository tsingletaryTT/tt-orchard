#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Check the Landlock sandbox of scripts/opencode_probe.py with real open, listdir, connect and bind calls, and no model.

    python3 scripts/landlock_selfcheck.py

Builds the same spec the runner builds for an attempt, runs a small Python program under scripts/landlock_exec.py, and prints, for each probe,
whether the sandboxed program got access. A probe that should be denied and was allowed makes the exit code 1. The program opens nothing for
writing outside the attempt directory except to test that the write fails, and connects only to ports on 127.0.0.1.

Expected: denied for the user's ssh key, Hugging Face token, GitHub and Docker credentials, the opencode config, the code directory, /tmp, /etc/shadow,
every Tenstorrent device node and another process's /proc/<pid>/environ; allowed for the attempt directory, its own /proc/self/environ and the model
server's port; refused for a connect to another port and for any bind.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import opencode_probe as op  # noqa: E402

PROGRAM = r'''
import json, os, socket, sys
H = os.environ["SELFCHECK_HOME"]
results = []
def probe(kind, path, should):
    try:
        if kind == "dir": os.listdir(path)
        elif kind == "dev": os.close(os.open(path, os.O_RDWR))
        elif kind == "write":
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL); os.close(fd); os.unlink(path)
        else: os.close(os.open(path, os.O_RDONLY))
        got = "allowed"
    except OSError as e:
        got = "denied"
    results.append({"probe": f"{kind} {path}", "expected": should, "got": got, "ok": got == should})
for kind, rel in (("file", ".ssh/id_ed25519"), ("file", ".cache/huggingface/token"), ("file", ".config/gh/hosts.yml"), ("file", ".docker/config.json"),
                  ("file", ".config/opencode/opencode.json"), ("dir", ".ssh"), ("dir", "code"), ("dir", "orchard-runs")):
    probe(kind, H + "/" + rel, "denied")
for kind, path in (("dir", "/tmp"), ("file", "/etc/shadow"), ("dir", "/dev/tenstorrent"), ("dev", "/dev/tenstorrent/0"), ("file", "/proc/1/environ"),
                   ("write", H + "/selfcheck-should-fail"), ("write", "/tmp/selfcheck-should-fail")):
    probe(kind, path, "denied")
probe("file", "/proc/self/environ", "allowed")
probe("write", os.getcwd() + "/selfcheck-ok", "allowed")
def tcp(port, should, bind=False):
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", 0)) if bind else s.settimeout(2) or s.connect(("127.0.0.1", port))
        got = "allowed"
    except PermissionError:
        got = "denied"
    except OSError:
        got = "allowed"        # refused or timed out by the network, not by the sandbox
    finally:
        s.close()
    results.append({"probe": ("bind" if bind else "connect") + f" 127.0.0.1:{port}", "expected": should, "got": got, "ok": got == should})
tcp(int(os.environ["SELFCHECK_PORT"]), "allowed")
tcp(22, "denied")
tcp(0, "denied", bind=True)
results.append({"probe": "environment names", "expected": "no tokens", "got": sorted(k for k in os.environ if any(w in k.upper() for w in ("TOKEN", "SECRET", "PASSW", "KEY"))) or "none", "ok": True})
print(json.dumps(results))
'''


def main() -> int:
    base = Path.home() / "orchard-runs" / "ocprobe" / "selfcheck"
    shutil.rmtree(base, ignore_errors=True)
    adir = base / "adir"
    (base / "meta" / "home").mkdir(parents=True)
    adir.mkdir(parents=True)
    (base / "meta" / "skill.md").write_text("x")
    port = 8011
    env = op.build_env(adir, base / "meta", None)
    env.update({"SELFCHECK_HOME": str(Path.home()), "SELFCHECK_PORT": str(port)})
    spec = op.sandbox_spec(adir, base / "meta", env, sim=None, connect=[port], argv=["python3", "-c", PROGRAM], bind=[])
    spec_path = base / "spec.json"
    spec_path.write_text(json.dumps(spec))
    r = subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "landlock_exec.py"), str(spec_path)], capture_output=True, text=True,
                       env={"PATH": "/usr/bin:/bin"})
    if r.returncode != 0:
        print("sandbox run failed:", r.stderr[:500])
        return 2
    rows = json.loads(r.stdout.strip().splitlines()[-1])
    bad = [x for x in rows if not x["ok"]]
    for x in rows:
        print(f"{'ok  ' if x['ok'] else 'FAIL'} {x['probe'].replace(str(Path.home()), '~'):60s} expected {x['expected']}, got {x['got']}")
    print(f"{len(rows) - len(bad)} of {len(rows)} probes as expected")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
