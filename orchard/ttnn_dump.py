# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`tt-orchard inventory refresh-ttnn`: record the names of the ops the installed ttnn offers.

ttnn's ops are bound from compiled code, so their names are in no Python file, and importing ttnn opens the device.
The import therefore runs under a one-chip gozer lease, once per ttnn version, and the names are written to
orchard/data/ttnn_ops-<version>-build-<build>.json (the build id is the date of the compiled library, so two builds that share a
version keep separate files; the loader in orchard/opinventory.py picks the dump for the installed build). The operator runs this
command. The supervisor never does.

With `--simulator PATH` (the ttsim library, for example libttsim_bh.so from the simulator image, with its
soc_descriptor.yaml beside it) the import runs against the simulator and takes no lease: the script only lists names and
never opens a device, and the simulator gives the import nothing to open.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import subprocess
from pathlib import Path

from orchard import hwtool, opinventory
from orchard.kernelprobe import _TICKET, GOZER_TERM_GRACE_S, cancel_ticket
from orchard.stallwatch import _kill_group

WAIT_AFTER_KILL_S = 10.0     # how long to read the output of a group that was stopped
MARK = "ORCHARD_TTNN_DUMP "
# Public callables at the top of ttnn, and every submodule of the namespaces named in WALK_ROOTS, to any depth (up to MAX_DEPTH).
# `experimental` has submodules of its own (experimental.deepseek_prefill.update_padded_kv_cache, issue 339), so one level is not enough.
# Other top-level ttnn submodules are listed under `unwalked_namespaces` and are not walked, so a missed namespace shows in the file.
WALK_ROOTS = ("experimental", "transformer")
MAX_DEPTH = 4
SCRIPT_TEMPLATE = r'''
import importlib.metadata as m, json, os, types
import ttnn

WALK_ROOTS = __WALK_ROOTS__
MAX_DEPTH = __MAX_DEPTH__
__BUILD_ID_SOURCE__

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


def is_ttnn_module(obj):
    # A module that ttnn itself defines. A namespace that imports numpy or torch must not pull their functions in.
    return isinstance(obj, types.ModuleType) and str(getattr(obj, "__name__", "")).startswith("ttnn")


names, unwalked = [], []


def walk(mod, prefix, depth, seen):
    for k in sorted(dir(mod)):
        if k.startswith("_"):
            continue
        label = prefix + "." + k
        obj = safe_get(mod, k, label)
        if is_op(obj):
            names.append(label)
        elif is_ttnn_module(obj) and depth < MAX_DEPTH and id(obj) not in seen:
            seen.add(id(obj))
            walk(obj, label, depth + 1, seen)


for n in sorted(dir(ttnn)):
    if n.startswith("_"):
        continue
    obj = safe_get(ttnn, n, n)
    if is_op(obj):
        names.append(n)
    elif isinstance(obj, types.ModuleType) and n in WALK_ROOTS:
        walk(obj, n, 1, {id(obj)})
    elif is_ttnn_module(obj):
        unwalked.append(n)

commit = next((v for v in (getattr(ttnn, a, None) for a in ("__git_hash__", "git_hash", "GIT_HASH")) if isinstance(v, str)), None)
try:
    build = build_id(os.path.dirname(ttnn.__file__), commit)
except Exception:
    build = None
# The leading newline keeps the marker at the start of its line even if ttnn printed text without one.
print("\n" + "ORCHARD_TTNN_DUMP " + json.dumps({"ttnn_version": m.version("ttnn"), "build": build, "names": names,
                                                   "skipped": skipped, "unwalked_namespaces": unwalked}))
'''
SCRIPT = (SCRIPT_TEMPLATE.replace("__WALK_ROOTS__", repr(WALK_ROOTS)).replace("__MAX_DEPTH__", str(MAX_DEPTH))
          .replace("__BUILD_ID_SOURCE__", inspect.getsource(opinventory.build_id)))


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
    """Write the dump with sorted, de-duplicated names and a sha256 of that list.

    The file is `ttnn_ops-<version>-build-<build>.json` when the dump names a build (the date of the compiled library, with a
    commit when one was found), so two builds that share a version never share a file (issue 339). A dump with no build id is
    written as `ttnn_ops-<version>.json`, and it is refused when that file already holds a different list of names: it may be
    another build."""
    version = dump["ttnn_version"]
    if not isinstance(version, str) or not opinventory.VERSION_RE.match(version):
        raise ValueError(f"ttnn version {version!r} is not a plain version token")
    build = dump.get("build")
    if build is not None and not (isinstance(build, str) and re.fullmatch(r"\d{8}(-[0-9a-f]{7,40})?|[0-9a-f]{7,40}", build)):
        raise ValueError(f"build id {build!r} is not a date (YYYYMMDD), a commit, or a date and a commit")
    names = sorted(set(dump["names"]))
    # sha256 = sha256 of json.dumps(names) with default separators, over the sorted, de-duplicated names.
    digest = hashlib.sha256(json.dumps(names).encode()).hexdigest()
    out = Path(data_dir) / (f"ttnn_ops-{version}-build-{build}.json" if build else f"ttnn_ops-{version}.json")
    if not build and out.is_file():
        try:
            held = json.loads(out.read_text(encoding="utf-8")).get("sha256")
        except (OSError, ValueError, AttributeError):
            held = None
        if held != digest:
            raise ValueError(f"{out.name} already holds a different list of names and this dump has no build id to name a "
                             "separate file; refusing to overwrite what may be another build")
    out.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps({"ttnn_version": version, **({"build": build} if build else {}), "names": names,
                       "skipped": sorted(set(dump.get("skipped", []))),
                       **({"unwalked_namespaces": sorted(set(dump["unwalked_namespaces"]))}
                          if dump.get("unwalked_namespaces") else {}), "sha256": digest}, indent=2) + "\n"
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
