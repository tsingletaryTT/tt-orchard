# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`tt-orchard inventory refresh-ttnn`: record the names of the ops the installed ttnn offers.

ttnn's ops are bound from compiled code, so their names are in no Python file, and importing ttnn opens the device.
The import therefore runs under a one-chip gozer lease, once per ttnn version, and the names are written to
orchard/data/ttnn_ops-<version>.json. The operator runs this command. The supervisor never does.

With `--simulator PATH` (the ttsim library, for example libttsim_bh.so from the simulator image, with its
soc_descriptor.yaml beside it) the import runs against the simulator and takes no lease: the script only lists names and
never opens a device, and the simulator gives the import nothing to open.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from orchard import hwtool, opinventory
from orchard.kernelprobe import _TICKET, GOZER_TERM_GRACE_S, cancel_ticket
from orchard.stallwatch import _kill_group

WAIT_AFTER_KILL_S = 10.0     # how long to read the output of a group that was stopped
MARK = "ORCHARD_TTNN_DUMP "
# Public callables at the top of ttnn and one level into the two namespaces that hold model ops.
SCRIPT = r'''
import importlib.metadata as m, json, types
import ttnn

_MISSING = object()
skipped = []


def safe_get(obj, name, label):
    """getattr that survives a lazy or deprecated attribute raising anything. The name is recorded in `skipped`."""
    try:
        return getattr(obj, name)
    except Exception:
        skipped.append(label)
        return _MISSING


def is_op(obj):
    # Classes (Tensor, MemoryConfig, ...) are types and are left out. Functions, builtins and callable objects are ops.
    return obj is not _MISSING and callable(obj) and not isinstance(obj, type)


names = []
for n in sorted(dir(ttnn)):
    if n.startswith("_"):
        continue
    obj = safe_get(ttnn, n, n)
    if is_op(obj):
        names.append(n)
    elif isinstance(obj, types.ModuleType) and n in ("experimental", "transformer"):
        for k in sorted(dir(obj)):
            if not k.startswith("_") and is_op(safe_get(obj, k, f"{n}.{k}")):
                names.append(f"{n}.{k}")
# The leading newline keeps the marker at the start of its line even if ttnn printed text without one.
print("\n" + "ORCHARD_TTNN_DUMP " + json.dumps({"ttnn_version": m.version("ttnn"), "names": names, "skipped": skipped}))
'''


SIMULATOR_ENV = {"TT_METAL_ARCH_NAME": "blackhole", "TT_METAL_SLOW_DISPATCH_MODE": "1"}


def dump_command(python, who: str = "claude:orchard-inventory", simulator=None) -> list[str]:
    """The full command line. It goes through `gozer run`, so the import holds a one-chip lease, unless `simulator`
    names a ttsim library: then it is `env TT_METAL_SIMULATOR=... python -c script` and no chip is involved."""
    if simulator:
        return ["env", f"TT_METAL_SIMULATOR={simulator}"] + [f"{k}={v}" for k, v in SIMULATOR_ENV.items()] + \
            [str(python), "-c", SCRIPT]
    return ["gozer", "run", "--chips", "1", "--exact", hwtool.probe_chip(), "--who", who,
            "--reason", "dump ttnn op names for tt-orchard inventory", "--", str(python), "-c", SCRIPT]


def parse(stdout: str) -> dict:
    """The dump the script printed on its marked line. Other output (logs, warnings) is ignored. The marker may follow
    banner text on its line, and when several lines carry it the last one wins."""
    found = None
    for line in stdout.splitlines():
        i = line.find(MARK)
        if i >= 0:
            found = line[i + len(MARK):]
    if found is None:
        raise ValueError("the dump printed no ORCHARD_TTNN_DUMP line")
    return json.loads(found)


def write_dump(dump: dict, data_dir) -> Path:
    """Write ttnn_ops-<version>.json with sorted, de-duplicated names and a sha256 of that list."""
    version = dump["ttnn_version"]
    if not isinstance(version, str) or not opinventory.VERSION_RE.match(version):
        raise ValueError(f"ttnn version {version!r} is not a plain version token")
    names = sorted(set(dump["names"]))
    out = Path(data_dir) / f"ttnn_ops-{version}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    # sha256 = sha256 of json.dumps(names) with default separators, over the sorted, de-duplicated names.
    body = json.dumps({"ttnn_version": version, "names": names, "skipped": sorted(set(dump.get("skipped", []))),
                       "sha256": hashlib.sha256(json.dumps(names).encode()).hexdigest()}, indent=2) + "\n"
    # Write beside the target and rename, so a crash never leaves a half-written file under the final name.
    tmp = out.with_name(out.name + ".tmp")
    try:
        tmp.write_text(body, encoding="utf-8")
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)
    return out


def _real_run(argv, timeout, term_grace_s: float = GOZER_TERM_GRACE_S):
    """Run `argv` in its own session. On a timeout the whole group gets SIGTERM, so `gozer run` forwards it and releases
    its lease (the release resets the chip, about 42 s), and SIGKILL only after `term_grace_s`. subprocess.run(timeout=)
    would SIGKILL gozer alone: its release would never run and the python child that opened the device would keep
    running. Returns (exit code, stdout, stderr); 127 when the command could not run or timed out."""
    try:
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                start_new_session=True)
    except OSError as exc:
        return 127, "", f"{type(exc).__name__}: {exc}"
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_group(proc, term_grace_s)
        try:
            out, err = proc.communicate(timeout=WAIT_AFTER_KILL_S)
        except subprocess.TimeoutExpired:
            out, err = "", ""
        return 127, out or "", f"timed out after {timeout} s and was stopped: {(err or '').strip()[-300:]}"
    except BaseException:
        _kill_group(proc, term_grace_s)           # an interrupt of the command must not leave the lease running
        raise
    return proc.returncode, out, err


def main(python, *, run=_real_run, data_dir=opinventory.DATA_DIR, say=print, cancel=cancel_ticket, simulator=None) -> int:
    """Show the command, run it, write the file. 0 on success, 2 when the dump failed or could not be read.
    With `simulator` the import runs against that ttsim library and no lease is taken."""
    if simulator and not os.path.isfile(simulator):
        say(f"the simulator library {simulator} does not exist")
        return 2
    argv = dump_command(python, simulator=simulator)
    if simulator:
        say("running against the simulator (no lease, no chip): " + " ".join(argv[:5]) + " -c <script>")
    else:
        say("running (it takes a one-chip lease): " + " ".join(argv[:12]) + " -c <script>")
    rc, out, err = run(argv, 1800)
    if rc == 10:
        # `gozer wait` is not suggested: it grants a lease that has no owner pid (see orchard/handoff.py).
        m = _TICKET.search(err + out)
        if m is None:
            say("gozer queued the lease (exit 10), so nothing ran; no ticket was found in its output, look with "
                "`gozer status`. Run the command again when chips are free")
        else:
            problem = cancel(m.group(1))
            say(f"gozer queued the lease (exit 10), so nothing ran; "
                + (f"the ticket {m.group(1)} was cancelled" if problem is None else
                   f"the ticket {m.group(1)} could not be cancelled: {problem}")
                + ". Run the command again when `gozer status` shows free chips")
        return 2
    if rc != 0:
        say(f"the dump failed (exit {rc}): {(err or out).strip()[-300:]}")
        return 2
    try:
        path = write_dump(parse(out), data_dir)
    except (ValueError, KeyError, OSError) as exc:
        say(f"the dump could not be read: {exc}")
        return 2
    say(f"wrote {path}")
    return 0
