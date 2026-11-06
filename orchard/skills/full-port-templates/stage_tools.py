# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Fixed-name tools for the coder of a full-port stage (issue 505). The supervisor copies this file to stages/<N>/stage_tools.py (next to
port_check.py) and the task text names it. Run it from anywhere:

    python3 stages/2/stage_tools.py last-check            which check failed, where, the numbers and the evidence file
    python3 stages/2/stage_tools.py layer-diff            the port against the CPU reference layer by layer: the first layer that diverges
    python3 stages/2/stage_tools.py facts [KEY]           one value of port_facts.json, by dotted path (config.hc_count); --find TEXT searches
    python3 stages/2/stage_tools.py show FILE --lines A:B a range of a file in the run directory, with line numbers
    python3 stages/2/stage_tools.py cheat [NAME]          one entry of the verified TTNN call sheet (issue 523); --sheet ttlang for the kernel sheet

    python3 stages/2/stage_tools.py edit FILE --old TEXT --new TEXT   change the one place where TEXT occurs in a file of stages/<N>/port (issue 524)

It reads only files in its own stage directory, and writes nothing there except what a tool says it writes (`edit` writes only below port/). It is standard library only: it must
run in the coder's shell, where the supervisor's own package is not importable. Everything it prints is advice built from files the coder could
also have edited; the supervisor's verdict is the gate's, taken from its own ledger. Each tool says what it read.

Why fixed names. In run 18 of the tiny Qwen4 rehearsal the coder spent its repair turns on one-off `cat | head`, `grep` and `python3 -c` lines. Each
was a new command for the watchdog to guess about. A tool call has a name and arguments, which the watchdog (orchard/watchdog.py `command_shape`)
counts exactly.
"""
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent               # stages/<N>, or stage 4's entry directory
OUTPUT_REL = "evidence/hw-test-output.txt"           # the supervisor's capture of the check's output
MARKER_REL = "test-result.json"                      # the supervisor's record of the test
RECORD_REL = "evidence/port-check.json"              # the check's own record
OUTPUT_MAX = 2_000_000                               # bytes read from the output file; larger is refused
JSON_MAX = 1_000_000
TOP1_MIN = 0.85                                      # the gate's bar (orchard/full_port.py TOP1_MIN; a test keeps the two equal)
FAILURE_TOKEN = "PORT_CHECK_FAILURE "                # starts the check's structured failure line (orchard/port_repair.py)
MAX_LINE = 300
ROWS = 8
LAYER_PORT_REL = "evidence/layer-port.npz"           # written by the worker of the hardware test from the port's optional `layer_outputs` hook
LAYER_REF_REL = "evidence/layer-reference.npz"       # the CPU reference's layers, cached here by `layer-diff` (with a sidecar that binds it to its inputs)
LAYER_REF_META_REL = "evidence/layer-reference.json"
LAYER_MARK = "PORT_CHECK_LAYERS"                     # the check's line that names the port's file: LAYER_MARK <sha256> <count>
LAYER_FILE_MAX = 2 ** 31                             # bytes: a larger layer file is refused before it is opened as an archive
MAX_LAYERS = 512
THRESHOLD = 0.05                                     # relative L2 error that counts as left the reference (the first_divergence probe's fallback)
LAYER_TIMEOUT_S = 3600                               # the reference run on the CPU
CONFIG_REL = "port_check_config.json"
TOOLS_CONFIG_REL = "stage_tools.json"                # written by the supervisor: {"reference_python": path} when it knows one
FACTS_REL = "port_facts.json"
PROMPT_REL = "stages/1/evidence/reference/prompt-ids.json"
ADVICE_NOTE = "advice from files in this directory; the gate's verdict is the supervisor's"


def open_dir_beneath(root, parts) -> int:
    """An open directory descriptor for root/parts[0]/parts[1]/..., each directory opened from the one before with O_NOFOLLOW, so a component that
    is a link (or was swapped for one after an earlier check) is refused with OSError (review of PR 506: O_NOFOLLOW on the leaf alone does not
    cover the directories above it). The caller closes the descriptor."""
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0))  # hardening: ok the run directory itself is the trusted anchor the supervisor named
    try:
        for part in parts:
            nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0), dir_fd=fd)
            os.close(fd)
            fd = nxt
        return fd
    except BaseException:
        os.close(fd)
        raise


def read_beneath(root, parts, cap: int):
    """(bytes, None) or (None, reason) for the regular file root/parts, with no link followed at any level."""
    try:
        dfd = open_dir_beneath(root, parts[:-1])
    except OSError as exc:
        return None, f"cannot be opened ({exc.__class__.__name__}: a link or an unreadable directory)"
    try:
        return read_file(Path(parts[-1]), cap, dir_fd=dfd)
    finally:
        os.close(dfd)


def read_file(path: Path, cap: int, dir_fd=None):
    """(bytes, None) or (None, reason). A regular file only: a link, a FIFO or a device is refused without being opened for blocking, and a file
    over `cap` is refused. The size is checked on the open descriptor, so a file swapped after a check is still bounded."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0), dir_fd=dir_fd)
    except FileNotFoundError:
        return None, "missing"
    except OSError as exc:
        return None, f"cannot be opened ({exc.__class__.__name__}: a link or an unreadable file)"
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            return None, "is not a regular file"
        if st.st_size > cap:
            return None, f"is larger than {cap} bytes"
        chunks, got = [], 0
        while got <= cap:                              # bounded: at most cap + 1 bytes, whatever the file grows to
            block = os.read(fd, min(1 << 20, cap + 1 - got))
            if not block:
                break
            chunks.append(block)
            got += len(block)
        data = b"".join(chunks)
        if len(data) > cap:
            return None, f"is larger than {cap} bytes"
        return data, None
    finally:
        os.close(fd)


def read_json(path: Path, expect=dict):
    """(value, None) or (None, reason): JSON of the expected type from a regular file."""
    data, why = read_file(path, JSON_MAX)
    if data is None:
        return None, why
    try:
        value = json.loads(data.decode("utf-8"))
    except (ValueError, RecursionError):
        return None, "is not JSON"
    if not isinstance(value, expect):
        return None, f"holds {type(value).__name__}; expected {expect.__name__}"
    return value, None


def new_file(path: Path):
    """A file object for a new regular file at `path`: any earlier file or link there is removed first, and the open refuses a link (O_EXCL, O_NOFOLLOW)."""
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0), 0o644)
    return os.fdopen(fd, "wb")  # hardening: ok the descriptor comes from os.open with O_EXCL and O_NOFOLLOW above


def clean(text, limit: int = MAX_LINE) -> str:
    """One printable line: control characters removed, cut to `limit`."""
    text = "".join(c if c.isprintable() else " " for c in str(text))
    return text if len(text) <= limit else text[:limit - 3] + "..."


def rel(path: Path) -> str:
    """The path as the task text names it: relative to the run directory when the stage sits under `stages/`."""
    parts = path.parts
    if "stages" in parts:
        return "/".join(parts[len(parts) - 1 - parts[::-1].index("stages"):])
    return str(path)


def whole(x) -> bool:
    return type(x) is int


def ids_ok(x) -> bool:
    return isinstance(x, list) and 0 < len(x) <= 100_000 and all(whole(i) and 0 <= i < 10**12 for i in x)


def failure_line(text: str):
    """The last well-formed structured failure, {"kind", "op"}, and the last `FAILED:` message, from the check's output."""
    found, message = None, None
    for line in text.splitlines():
        if line.startswith("FAILED: "):
            message = line[len("FAILED: "):]
        if line.startswith(FAILURE_TOKEN):
            try:
                doc = json.loads(line[len(FAILURE_TOKEN):])
            except ValueError:
                continue
            if isinstance(doc, dict) and doc.get("kind") in ("port", "setup") and (doc.get("op") is None or isinstance(doc.get("op"), str)):
                found = doc
    return found, message


def where(position: int) -> str:
    return ("the token after the prompt: the prefill logits are wrong" if position == 0 else
            "a decode step: the prefill token matched, so look at the decode path, the cache position and the state updates")


def last_check(stage: Path) -> list[str]:
    """The summary of the last check in `stage`, as lines (fewer than 40)."""
    stage = Path(stage)
    lines: list[str] = []
    marker, why = read_json(stage / MARKER_REL)
    if marker is None:
        lines.append("no check has run in this directory yet." if why == "missing" else f"not checked: {MARKER_REL} {why}.")
        return lines
    code = marker.get("returncode")
    if not whole(code):
        return [f"not checked: {MARKER_REL} has no whole-number returncode."]
    command = marker.get("command") if isinstance(marker.get("command"), str) else "?"
    lines.append(f"last check: {clean(command, 100)} (exit {code}{', timed out' if marker.get('timed_out') is True else ''})")
    data, why = read_file(stage / OUTPUT_REL, OUTPUT_MAX)
    text = data.decode("utf-8", "replace") if data is not None else ""
    if data is None:
        lines.append(f"not checked: {OUTPUT_REL} {why}.")
    else:
        out = marker.get("output")
        want = out.get("sha256") if isinstance(out, dict) else None
        if want != hashlib.sha256(data).hexdigest():
            lines.append(f"warning: {OUTPUT_REL} does not match the sha256 the supervisor recorded for it; it may have changed since the test.")
    if code == 0 and marker.get("timed_out") is not True:
        lines += numerics(stage)
    elif data is not None:
        kind_op, message = failure_line(text)
        if kind_op is not None:
            kind, op = kind_op["kind"], kind_op["op"]
            lines.append(f"result: FAILED, kind {kind}" + (f", at the '{op}' request" if op else ", before any request reached the port"))
            if kind == "setup":
                lines.append("  the run around the port was wrong (chips, stage 1 files, runtime). An edit to the port cannot fix it.")
        else:
            lines.append("result: FAILED, with no structured failure line (a crash or a timeout before the check could say what failed).")
        lines.append("message: " + clean(message if message is not None else "(none printed)"))
        lines.append(f"evidence: {rel(stage / OUTPUT_REL)}")
    lines.append(f"({ADVICE_NOTE})")
    return lines


def numerics(stage: Path) -> list[str]:
    """The numbers of a check that ran to the end, from its record: recomputed from the two id lists."""
    record, why = read_json(stage / RECORD_REL)
    path = rel(stage / RECORD_REL)
    if record is None:
        return [f"not checked: {RECORD_REL} {why}, so the agreement cannot be read."]
    ref, chip = record.get("reference_ids"), record.get("chip_ids")
    if not ids_ok(ref) or not ids_ok(chip) or len(ref) != len(chip):
        return [f"not checked: {RECORD_REL} has no two equal-length lists of whole-number ids."]
    n = len(ref)
    rows = [(i, a, b) for i, (a, b) in enumerate(zip(ref, chip)) if a != b]
    agreement = (n - len(rows)) / n
    lines = []
    verdict = "meets" if agreement >= TOP1_MIN else "is BELOW"
    lines.append(f"result: ran to the end. top1_agreement {agreement:.4f} ({n - len(rows)} of {n} ids match) {verdict} the gate's {TOP1_MIN}.")
    stated = record.get("top1_agreement")
    if isinstance(stated, (int, float)) and not isinstance(stated, bool) and abs(stated - agreement) > 1e-6:
        lines.append(f"warning: the record's own top1_agreement {stated} differs from the ids' {agreement:.4f}; the ids are used.")
    if rows:
        lines.append(f"{len(rows)} of {n} ids differ; first mismatch at position {rows[0][0]}: {where(rows[0][0])}.")
        lines.append("position | reference id | port id")
        lines += [f"{p} | {a} | {b}" for p, a, b in rows[:ROWS]]
        if len(rows) > ROWS:
            lines.append(f"({len(rows) - ROWS} more not listed)")
        lines.append("this output names no layer or feature; `layer-diff` compares the port with the reference layer by layer.")
    lines.append(f"evidence: {path}")
    return lines


# ---- layer-diff --------------------------------------------------------------------------------------------------------------------
# The outer part (standard library only) checks that the port's layer file is the one the supervisor's captured output names, finds the reference
# interpreter and runs the inner part under it. The inner part needs numpy, and torch with transformers when the reference is not cached.

def file_sha256(path: Path, cap: int, dir_fd=None):
    """(hex digest, None) or (None, reason) of a regular file, read in blocks from the open descriptor (no link followed, no FIFO opened)."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0), dir_fd=dir_fd)
    except FileNotFoundError:
        return None, "missing"
    except OSError as exc:
        return None, f"cannot be opened ({exc.__class__.__name__}: a link or an unreadable file)"
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            return None, "is not a regular file"
        if st.st_size > cap:
            return None, f"is larger than {cap} bytes"
        digest, seen = hashlib.sha256(), 0
        while True:
            block = os.read(fd, 1 << 20)
            if not block:
                return digest.hexdigest(), None
            seen += len(block)
            if seen > cap:
                return None, f"is larger than {cap} bytes"
            digest.update(block)
    finally:
        os.close(fd)


def run_dir_of(stage: Path) -> Path:
    for p in stage.parents:
        if p.name == "stages":
            return p.parent
    return stage.parent


def sealed_layers_line(stage: Path):
    """((sha256, count), None) from the supervisor's capture of the check's output, or (None, why). The capture must match the sha256 in the
    supervisor's own test record, and must hold exactly one LAYER_MARK line: only the check prints that line (it prefixes every line the port
    prints), so a layer file named there is the file the hardware test wrote."""
    marker, why = read_json(stage / MARKER_REL)
    if marker is None:
        return None, f"{MARKER_REL} {why}: no check has run in this directory yet"
    data, why = read_file(stage / OUTPUT_REL, OUTPUT_MAX)
    if data is None:
        return None, f"{OUTPUT_REL} {why}"
    out = marker.get("output")
    if not isinstance(out, dict) or out.get("sha256") != hashlib.sha256(data).hexdigest():
        return None, f"{OUTPUT_REL} does not match the sha256 the supervisor recorded for it, so it may have been edited"
    text = data.decode("utf-8", "replace")
    found = re.findall(rf"^{LAYER_MARK} ([0-9a-f]{{64}}) (\d{{1,6}})\s*$", text, re.M)
    starts = [l for l in text.splitlines() if l.startswith(LAYER_MARK)]
    if len(starts) > 1:
        return None, f"{OUTPUT_REL} has {len(starts)} {LAYER_MARK} lines; the check prints one, so something else wrote a line of its own"
    if not found:
        if (stage / LAYER_PORT_REL).exists() or (stage / LAYER_PORT_REL).is_symlink():
            return None, f"{LAYER_PORT_REL} exists but the output has no {LAYER_MARK} line naming it: it was not written by the last check"
        return None, "none"
    return (found[0][0], int(found[0][1])), None


def reference_python(stage: Path, given):
    """The interpreter that has torch and transformers: `--python`, else `reference_python` in stage_tools.json, else $ORCHARD_REFERENCE_PYTHON."""
    config, _ = read_json(stage / TOOLS_CONFIG_REL)
    for candidate in (given, config.get("reference_python") if config else None, os.environ.get("ORCHARD_REFERENCE_PYTHON")):
        if isinstance(candidate, str) and candidate and "\0" not in candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def threshold_arg(text):
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and 0 < value <= 1e6 else None


def layer_diff(stage: Path, argv: list) -> int:
    import argparse
    parser = argparse.ArgumentParser(prog="stage_tools.py layer-diff", add_help=True)
    parser.add_argument("--threshold", default=str(THRESHOLD), help=f"relative L2 error that counts as diverged (default {THRESHOLD})")
    parser.add_argument("--python", default=None, help="interpreter with numpy, torch and transformers (default: stage_tools.json, then $ORCHARD_REFERENCE_PYTHON)")
    parser.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"), help="the reference's dtype on the CPU (bfloat16 halves the memory)")
    args = parser.parse_args(argv)
    if threshold_arg(args.threshold) is None:
        print(f"refused: --threshold must be a number above 0 and at most 1e6, got {clean(args.threshold, 40)!r}")
        return 2
    sealed, why = sealed_layers_line(stage)
    if sealed is None:
        if why == "none":
            print("no layer outputs from the last check: either the port module has no `layer_outputs(model, ids)` (see the task text for what it returns), "
                  f"or the last check ended at or above {TOP1_MIN} agreement (the hook runs only on a numerics failure), or it failed before the comparison.")
        else:
            print("refused: " + why + ".")
        return 2
    digest, why = file_sha256(stage / LAYER_PORT_REL, LAYER_FILE_MAX)
    if digest is None:
        print(f"refused: {LAYER_PORT_REL} {why}.")
        return 2
    if digest != sealed[0]:
        print(f"refused: {LAYER_PORT_REL} is not the file the test wrote (its sha256 differs from the one in the captured output).")
        return 2
    python = reference_python(stage, args.python)
    if python is None:
        print("refused: no reference interpreter. Pass --python PATH (an interpreter with numpy, torch and transformers), or set ORCHARD_REFERENCE_PYTHON.")
        return 2
    cmd = [python, "-I", str(Path(__file__).resolve()), "--layer-diff-inner", args.threshold, args.dtype, digest, str(sealed[1])]
    env = {"PATH": os.environ.get("PATH", ""), "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
    try:
        done = subprocess.run(cmd, cwd=str(stage), env=env, capture_output=True, text=True, timeout=LAYER_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        print(f"refused: the reference run did not finish in {LAYER_TIMEOUT_S} s.")
        return 2
    except OSError as exc:
        print(f"refused: the reference interpreter could not start ({exc.__class__.__name__}).")
        return 2
    print("\n".join(clean(l, 400) for l in done.stdout.splitlines()[:60]))
    if done.returncode not in (0, 1, 2):
        print("the comparison stopped unexpectedly: " + clean(" ".join(done.stderr.strip().splitlines()[-2:]), 300))
        return 2
    return done.returncode


# ---- the inner part: numpy, and torch when the reference has to be computed ------------------------------------------------------------

def load_layers(path: Path, what: str):
    """({name: array}, None) or (None, reason) from an .npz of float arrays. Pickled arrays are refused (`allow_pickle=False`), and the member count is bounded."""
    import numpy as np
    try:
        with np.load(path, allow_pickle=False) as z:
            names = list(z.files)
            if len(names) > MAX_LAYERS + 1:
                return None, f"{what} holds more than {MAX_LAYERS} arrays"
            out = {n: np.array(z[n]) for n in names if re.fullmatch(r"layer_\d{3}|prompt_ids", n)}
    except Exception as exc:               # a corrupt archive, a pickled array, a bad header: one reason, not a traceback
        return None, f"{what} cannot be read ({exc.__class__.__name__})"
    return out, None


def ordered(arrays: dict, what: str):
    """The layers of a loaded file in order, or a reason: the names must be layer_000, layer_001, ... with no gap, and the arrays numeric."""
    names = sorted(n for n in arrays if n.startswith("layer_"))
    if not names or names != [f"layer_{k:03d}" for k in range(len(names))]:
        return None, f"{what} has no layers named layer_000, layer_001, ... in order"
    if any(arrays[n].dtype.kind not in "fiub" for n in names):
        return None, f"{what} holds an array that is not numeric"
    return [arrays[n] for n in names], None


def compare_layers(ref: list, port: list, threshold: float) -> dict:
    """Per layer: relative L2 error ||port - ref|| / ||ref|| (in float64), the cosine, both shapes, and whether the values are finite. `first` is the index
    of the first layer that is over `threshold`, has non-finite values, or has another number of elements; None when none is."""
    import numpy as np
    rows, first = [], None
    for k in range(min(len(ref), len(port))):
        a, b = ref[k], port[k]
        row = {"layer": k, "ref_shape": tuple(a.shape), "port_shape": tuple(b.shape), "rel": None, "cos": None, "bad": None}
        if a.size != b.size:
            row["bad"] = f"size differs: {a.size} elements in the reference, {b.size} in the port"
        else:
            x, y = np.asarray(a, dtype=np.float64).reshape(-1), np.asarray(b, dtype=np.float64).reshape(-1)
            if not np.isfinite(y).all():
                row["bad"] = f"non-finite values in the port ({int((~np.isfinite(y)).sum())} of {y.size})"
            elif not np.isfinite(x).all():
                row["bad"] = "non-finite values in the reference"
            else:
                nx, ny, diff = float(np.linalg.norm(x)), float(np.linalg.norm(y)), float(np.linalg.norm(x - y))
                row["rel"] = diff / nx if nx > 0 else diff
                row["cos"] = float(x @ y / (nx * ny)) if nx > 0 and ny > 0 else None
                if row["rel"] > threshold:
                    row["bad"] = f"relative error {row['rel']:.4f} is over {threshold}"
        rows.append(row)
        if row["bad"] and first is None:
            first = k
    return {"rows": rows, "first": first, "n_ref": len(ref), "n_port": len(port)}


def layer_kind(k: int):
    """The kind of layer k from the facts file's `layer_schedule` ({"linear_attention": "0-2", "full_attention": "3"}), or None."""
    facts, _ = read_json(HERE / FACTS_REL)
    schedule = facts.get("layer_schedule") if facts else None
    if not isinstance(schedule, dict):
        return None
    for kind, spec in list(schedule.items())[:20]:
        if not isinstance(kind, str) or not isinstance(spec, str) or len(spec) > 200:
            continue
        for part in spec.split(","):
            m = re.fullmatch(r"\s*(\d{1,6})(?:\s*-\s*(\d{1,6}))?\s*", part)
            if m and int(m.group(1)) <= k <= int(m.group(2) or m.group(1)):
                return clean(kind, 60)
    return None


def render(result: dict, threshold: float, notes: list) -> list:
    rows, first = result["rows"], result["first"]
    lines = [clean(n) for n in notes]
    if result["n_port"] != result["n_ref"]:
        lines.append(f"the port returned {result['n_port']} layers and the reference has {result['n_ref']}: compared as far as both go.")
    lines.append("layer | relative error | cosine | shapes (reference / port)")
    if first is not None:
        show = range(max(0, first - 3), min(len(rows), first + 5))
    else:
        show = sorted(set(range(min(3, len(rows)))) | set(range(max(0, len(rows) - 3), len(rows))))
    prev = None
    for k in show:
        if prev is not None and k != prev + 1:
            lines.append("  ...")
        r = rows[k]
        rel = "n/a" if r["rel"] is None else f"{r['rel']:.4f}"
        cos = "n/a" if r["cos"] is None else f"{r['cos']:.4f}"
        shapes = f"{r['ref_shape']} / {r['port_shape']}" if r["ref_shape"] != r["port_shape"] else f"{r['ref_shape']}"
        lines.append(f"{k:>5} | {rel} | {cos} | {shapes}" + ("   <- first over" if k == first else ""))
        prev = k
    if first is None:
        lines.append(f"all {len(rows)} layers are within a relative error of {threshold} of the reference on this prompt. The difference is outside the layers "
                     "(the final norm, the output head, the sampling) or only appears in decode (the cache, the positions): these are prefill layer outputs.")
    else:
        r = rows[first]
        kind = layer_kind(first)
        before = "" if first == 0 or rows[first - 1]["rel"] is None else f" (layer {first - 1} was {rows[first - 1]['rel']:.4f})"
        lines.append(f"first layer over the threshold: layer {first}: {r['bad']}{before}.")
        if kind:
            lines.append(f"layer {first} is {kind} in the layer schedule of port_facts.json: look at what that layer type does that earlier layers do not.")
    return lines


def reference_layers(snapshot: str, ids: list, dtype: str) -> list:
    """The CPU reference's output of each decoder layer for the prompt `ids`, as float32 numpy arrays, one per layer. Forward hooks on the decoder layers
    capture the raw layer outputs (transformers' own `hidden_states` ends with the final norm applied, so its last entry is not a layer output)."""
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM
    want = getattr(torch, dtype)
    try:
        model = AutoModelForCausalLM.from_pretrained(snapshot, dtype=want, local_files_only=True)
    except TypeError:                                  # an older transformers names the argument torch_dtype
        model = AutoModelForCausalLM.from_pretrained(snapshot, torch_dtype=want, local_files_only=True)
    model.eval()
    stack = None
    for path in ("model.layers", "layers", "model.model.layers", "transformer.h", "model.decoder.layers"):
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "__len__") and len(obj):
            stack = obj
            break
    if stack is None:
        raise RuntimeError("no decoder layer list found (tried model.layers, layers, transformer.h, model.decoder.layers)")
    captured = []

    def hook(_module, _inputs, output):
        out = output[0] if isinstance(output, (tuple, list)) else output
        captured.append(out.detach().float().cpu().numpy().astype(np.float32))

    handles = [layer.register_forward_hook(hook) for layer in stack]
    try:
        with torch.no_grad():
            model(input_ids=torch.tensor([ids]), use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return captured


def prompt_ids(run_dir: Path):
    doc, why = read_json(run_dir / PROMPT_REL)
    ids = doc.get("prompt_ids") if doc else None
    if not ids_ok(ids):
        return None, f"{PROMPT_REL} {why or 'holds no list of whole numbers under prompt_ids'}"
    return ids, None


def cached_reference(stage: Path, key: dict):
    """The cached reference layers when the sidecar names this prompt, snapshot and dtype and the npz is the file it hashed, else None."""
    meta, _ = read_json(stage / LAYER_REF_META_REL)
    if not meta or any(meta.get(k) != v for k, v in key.items()):
        return None
    digest, _ = file_sha256(stage / LAYER_REF_REL, LAYER_FILE_MAX)
    if digest is None or meta.get("npz_sha256") != digest:
        return None
    arrays, _ = load_layers(stage / LAYER_REF_REL, LAYER_REF_REL)
    layers, _ = ordered(arrays, LAYER_REF_REL) if arrays is not None else (None, None)
    return layers


def store_reference_cache(stage: Path, key: dict, ref: list, np) -> None:
    """Write the reference cache (npz, then the sidecar that binds it to `key`). Every create, rename and removal goes through a descriptor for the
    evidence directory that was opened from the run with O_NOFOLLOW at each level, so a linked `evidence` or `stages` directory cannot send a
    write or a removal outside the run (review of PR 506). Raises OSError when the directory is not reachable that way."""
    run = run_dir_of(stage)
    dfd = open_dir_beneath(str(run), Path(stage).relative_to(run).parts + (os.path.dirname(LAYER_REF_REL),))
    try:
        def write(name: str, produce):
            tmp = name + ".tmp"
            try:
                os.unlink(tmp, dir_fd=dfd)
            except FileNotFoundError:
                pass
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0), 0o644, dir_fd=dfd)
            with os.fdopen(fd, "wb") as f:  # hardening: ok the descriptor comes from os.open with O_EXCL and O_NOFOLLOW, beneath a directory descriptor
                produce(f)
            os.replace(tmp, name, src_dir_fd=dfd, dst_dir_fd=dfd)
        npz, side = os.path.basename(LAYER_REF_REL), os.path.basename(LAYER_REF_META_REL)
        write(npz, lambda f: np.savez(f, **{f"layer_{k:03d}": a for k, a in enumerate(ref)}))
        digest, _ = file_sha256(Path(npz), LAYER_FILE_MAX, dir_fd=dfd)
        write(side, lambda f: f.write(json.dumps({**key, "npz_sha256": digest, "layers": len(ref)}).encode()))
    finally:
        os.close(dfd)


def inner(argv: list) -> int:
    import numpy as np
    threshold, dtype, digest, count = float(argv[0]), argv[1], argv[2], int(argv[3])
    stage = HERE
    run_dir = run_dir_of(stage)
    arrays, why = load_layers(stage / LAYER_PORT_REL, LAYER_PORT_REL)
    if arrays is None:
        print("refused: " + why + ".")
        return 2
    port, why = ordered(arrays, LAYER_PORT_REL)
    if port is None:
        print("refused: " + why + ".")
        return 2
    ids, why = prompt_ids(run_dir)
    if ids is None:
        print("refused: " + why + ".")
        return 2
    if "prompt_ids" not in arrays or [int(i) for i in arrays["prompt_ids"].reshape(-1)] != ids:
        print("refused: the port's layer file was computed on a prompt that is not the stage 1 prompt, so the two cannot be compared.")
        return 2
    config, why = read_json(stage / CONFIG_REL)
    snapshot = config.get("model_snapshot") if config else None
    if not isinstance(snapshot, str) or not snapshot or "\0" in snapshot:
        print(f"refused: {CONFIG_REL} {why or 'names no model_snapshot'}.")
        return 2
    key = {"prompt_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(), "snapshot": snapshot, "dtype": dtype}
    ref = cached_reference(stage, key)
    source = "cached reference"
    if ref is None:
        try:
            ref = reference_layers(snapshot, ids, dtype)
        except BaseException as exc:         # an import that is missing, a model the installed transformers does not know, no memory: a reason
            print(f"cannot compute the reference: {exc.__class__.__name__}: {clean(str(exc), 250)}")
            return 2
        source = f"computed now on the CPU ({dtype})"
        try:                                  # the cache: a temporary file, a rename, and a sidecar that binds it to the prompt, the snapshot and the dtype
            store_reference_cache(stage, key, ref, np)
        except OSError:
            pass                              # the cache is a convenience
    result = compare_layers(ref, port, threshold)
    notes = [f"layer-diff: the port's {count} layer outputs against the CPU reference on the {len(ids)}-token stage 1 prompt; threshold {threshold}.",
             f"reference: {len(ref)} layers, {source}. port file {LAYER_PORT_REL}, sha256 {digest[:12]}, the file the check's output names.",
             f"({ADVICE_NOTE})"]
    print("\n".join(render(result, threshold, notes)))
    return 0 if result["first"] is None else 1


# ---- facts and show ---------------------------------------------------------------------------------------------------------------

VALUE_CHARS = 3000                 # characters of one value `facts` prints
FIND_HITS = 30
FIND_NODES = 200_000               # nodes visited by --find: a hostile file cannot make it run long
SHOW_DEFAULT = 200
SHOW_MAX = 400                     # lines in one `show`
SHOW_FILE_MAX = 1_000_000


def facts_cmd(stage: Path, argv: list):
    """(exit code, lines) for `facts [KEY] [--find TEXT]` over the stage's port_facts.json."""
    import argparse
    parser = argparse.ArgumentParser(prog="stage_tools.py facts", add_help=True)
    parser.add_argument("key", nargs="?", default=None, help="dotted path, for example config.hc_count or tensors.0")
    parser.add_argument("--find", default=None, help="list every key or text that contains this (case-insensitive)")
    args = parser.parse_args(argv)
    facts, why = read_json(stage / FACTS_REL)
    if facts is None:
        return 2, [f"{FACTS_REL} {why}."]
    if args.find is not None:
        needle = args.find.lower()
        if not needle or len(needle) > 100 or "\0" in needle:
            return 2, ["--find needs 1 to 100 characters of text."]
        hits, extra, seen, todo = [], 0, 0, [("", facts)]
        while todo and seen < FIND_NODES:
            path, node = todo.pop()
            seen += 1
            if isinstance(node, dict):
                kids = [(f"{path}.{k}" if path else str(k), v) for k, v in node.items()]
            elif isinstance(node, list):
                kids = [(f"{path}.{i}" if path else str(i), v) for i, v in enumerate(node)]
            else:
                kids = []
            for kid_path, v in reversed(kids):
                todo.append((kid_path, v))
            if path and (needle in path.rsplit(".", 1)[-1].lower() or (not isinstance(node, (dict, list)) and needle in str(node).lower())):
                if len(hits) < FIND_HITS:
                    hits.append(f"{clean(path, 120)}: {clean(json.dumps(node)[:150], 160)}")
                else:
                    extra += 1
        if not hits:
            return 1, [f"no key or text contains {clean(args.find, 60)!r}."]
        return 0, hits + ([f"({extra} more matches not shown; narrow the text)"] if extra else [])
    if args.key is None:
        return 0, [f"{k}: {describe(v)}" for k, v in list(facts.items())[:40]] + [f"(`facts KEY` prints one; `facts --find TEXT` searches; {ADVICE_NOTE})"]
    key = args.key
    if not key or len(key) > 200 or "\0" in key or any(part == "" for part in key.split(".")):
        return 2, ["the key is a dotted path such as config.hc_count: no empty parts, at most 200 characters."]
    node, walked = facts, []
    for part in key.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        elif isinstance(node, list) and re.fullmatch(r"\d{1,9}", part) and int(part) < len(node):
            node = node[int(part)]
        else:
            there = list(node.keys())[:20] if isinstance(node, dict) else ([f"0..{len(node) - 1}"] if isinstance(node, list) and node else [])
            return 1, [f"no key {clean(part, 60)!r} in {'.'.join(walked) or 'the file'}" + (f"; the keys are: {clean(', '.join(map(str, there)), 400)}" if there else "")]
        walked.append(part)
    compact = json.dumps(node)
    text = compact if len(compact) <= 100 else json.dumps(node, indent=1)
    if len(text) > VALUE_CHARS:
        text = text[:VALUE_CHARS] + f"\n... ({len(text) - VALUE_CHARS} more characters not shown; ask for a deeper key)"
    return 0, text.splitlines()


def describe(value) -> str:
    if isinstance(value, dict):
        return f"object with {len(value)} keys"
    if isinstance(value, list):
        return f"list of {len(value)}"
    return clean(json.dumps(value), 80)


def show_cmd(stage: Path, argv: list):
    """(exit code, lines) for `show FILE [--lines A:B]`: a range of a regular file inside the run directory, with line numbers."""
    import argparse
    parser = argparse.ArgumentParser(prog="stage_tools.py show", add_help=True)
    parser.add_argument("file", help="a path relative to this stage directory or to the run directory")
    parser.add_argument("--lines", default=f"1:{SHOW_DEFAULT}", help=f"first:last, 1-based, at most {SHOW_MAX} lines (default 1:{SHOW_DEFAULT})")
    args = parser.parse_args(argv)
    m = re.fullmatch(r"(\d{1,9}):(\d{1,9})", args.lines)
    if not m or int(m.group(1)) < 1 or int(m.group(2)) < int(m.group(1)):
        return 2, ["--lines must be FIRST:LAST with 1 <= FIRST <= LAST, for example --lines 10:80."]
    first, last = int(m.group(1)), int(m.group(2))
    if last - first + 1 > SHOW_MAX:
        return 2, [f"--lines: at most {SHOW_MAX} lines at a time; ask for the next range."]
    if not args.file or "\0" in args.file or len(args.file) > 500:
        return 2, ["the file name is empty, too long or has a NUL."]
    run = os.path.realpath(run_dir_of(stage))
    target = None
    for base in (stage, Path(run)):
        candidate = os.path.realpath(os.path.join(base, args.file))
        if candidate == run or not candidate.startswith(run + os.sep):
            return 2, [f"{clean(args.file, 100)} is outside the run directory: `show` reads only files in the run. The reference classes are in reference_excerpts.md."]
        if os.path.lexists(candidate):
            target = candidate
            break
    if target is None:
        return 2, [f"{clean(args.file, 100)} missing (looked in the stage directory and the run directory)."]
    data, why = read_beneath(run, Path(os.path.relpath(target, run)).parts, SHOW_FILE_MAX)   # each directory opened from the run with O_NOFOLLOW
    if data is None:
        return 2, [f"{clean(args.file, 100)} {why}."]
    if b"\0" in data[:8192]:
        return 2, [f"{clean(args.file, 100)} is a binary file."]
    rows = data.decode("utf-8", "replace").splitlines()
    if first > len(rows):
        return 2, [f"{clean(args.file, 100)} has {len(rows)} lines; --lines starts at {first}."]
    last = min(last, len(rows))
    head = f"{os.path.relpath(target, run)} ({len(rows)} lines), lines {first}-{last}"
    return 0, [head] + [f"{n:>6}  {clean(rows[n - 1])}" for n in range(first, last + 1)]


CHEAT_FILES = {"ttnn": "cheat_sheet.json", "ttlang": "ttlang_cheat_sheet.json"}      # the supervisor's copies of the verified sheets (issue 523)
CHEAT_LINE = 700


def cheat_cmd(stage: Path, argv: list):
    """(exit code, lines) for `cheat [NAME] [--sheet ttnn|ttlang]`: one entry of a verified cheat sheet, or the names when there is no NAME.
    The sheet is the supervisor's file in this directory (orchard/op_cheats.py checked it against the source tree); a name matches an entry's
    full name or its last part when only one entry has it."""
    import argparse
    parser = argparse.ArgumentParser(prog="stage_tools.py cheat", add_help=True)
    parser.add_argument("name", nargs="?", default=None, help="an entry name, for example ttnn.linear")
    parser.add_argument("--sheet", choices=sorted(CHEAT_FILES), default="ttnn")
    args = parser.parse_args(argv)
    sheet, why = read_json(stage / CHEAT_FILES[args.sheet])
    if sheet is None:
        return 2, [f"{CHEAT_FILES[args.sheet]} {why}."]
    # The installed sheet is coder-writable: test each container type before iterating (a null `sections` or `entries` is a diagnostic, not a traceback).
    sections = sheet.get("sections", [])
    if not isinstance(sections, list) or any(isinstance(sec, dict) and not isinstance(sec.get("entries", []), list) for sec in sections):
        return 2, [f"{CHEAT_FILES[args.sheet]} is malformed: `sections` and each section's `entries` must be lists."]
    found = [e for sec in sections if isinstance(sec, dict) for e in sec.get("entries", []) if isinstance(e, dict)]
    if args.name is None:
        return 0, [f"{len(found)} entries in the {args.sheet} sheet; `cheat NAME` prints one:"] + [clean(", ".join(str(e.get("name")) for e in found[i:i + 6]), CHEAT_LINE)
                                                                                                 for i in range(0, len(found), 6)]
    if not args.name or len(args.name) > 120 or "\0" in args.name:
        return 2, ["NAME is 1 to 120 characters."]
    exact = [e for e in found if e.get("name") == args.name]
    short = [e for e in found if str(e.get("name")).rsplit(".", 1)[-1] == args.name]
    hits = exact or short or [e for e in found if args.name.lower() in str(e.get("name")).lower()]
    if not hits:
        return 1, [f"no entry in the {args.sheet} sheet matches {clean(args.name, 60)!r}; `cheat` alone lists the names."]
    out = []
    for e in hits[:3]:
        out += [clean(f"{e.get('signature')}" + (f" [{e['status']}]" if e.get("status") else ""), CHEAT_LINE), *["    " + clean(ln, CHEAT_LINE) for ln in str(e.get("usage")).split("\n")[:12]],
                "    " + clean(e.get("notes"), CHEAT_LINE), "    " + clean(f"seen: {e.get('seen_in')}", CHEAT_LINE)]
    return 0, out + ([f"({len(hits) - 3} more entries match; give the full name)"] if len(hits) > 3 else [])

# ---- edit -------------------------------------------------------------------------------------------------------------------------

EDIT_MAX_BYTES = 400_000           # the most a port file may hold after an edit: the agent's own write bound (orchard/runaway.py MAX_FILE_BYTES; a test keeps the two equal)
EDIT_MAX_LINES = 6_000             # and its line bound (runaway.MAX_FILE_LINES)
EDIT_TEXT_MAX = 200_000            # bytes read from an --old-file or --new-file
EDIT_SHOWN = 3                     # closest lines shown when the old text is not found
EDIT_PLACES = 9                    # matches listed when the old text occurs more than once
PORT_DIR = "port"                  # the directory of the stage that `edit` may change
RECORD_MARK = "EDIT_RECORD "       # starts the one JSON line a successful call prints; the transcript keeps it


class _EditArgs(Exception):
    pass


def _edit_parser():
    import argparse

    class Parser(argparse.ArgumentParser):
        def error(self, message):
            raise _EditArgs(message)
    parser = Parser(prog="stage_tools.py edit", add_help=False)
    parser.add_argument("file")
    parser.add_argument("--old", default=None)
    parser.add_argument("--new", default=None)
    parser.add_argument("--old-file", default=None)
    parser.add_argument("--new-file", default=None)
    return parser


def _inside(base, candidate: str) -> bool:
    return candidate.startswith(str(base).rstrip(os.sep) + os.sep)


def port_target(stage: Path, name: str):
    """(parts below the stage directory, None) for the port file `name` names, or (None, reason). The name is taken from the run directory, from the stage
    directory or from the port directory, as `show` takes names. The path is cleaned of `.` and `..` first and must then lie below <stage>/port; the file is
    opened later, one directory at a time with O_NOFOLLOW, so a link anywhere on the way is refused."""
    if not name or "\0" in name or len(name) > 500:
        return None, "the file name is empty, too long or has a NUL."
    port_dir = os.path.join(stage, PORT_DIR)
    run = run_dir_of(stage)
    candidates = [os.path.normpath(os.path.join(run, name)), os.path.normpath(os.path.join(stage, name)), os.path.normpath(os.path.join(port_dir, name))]
    for i, cand in enumerate(candidates):
        if _inside(port_dir, cand) and (i < 2 or os.path.lexists(cand)):
            return Path(os.path.relpath(cand, stage)).parts, None
    return None, (f"{clean(name, 100)} is not in the port directory ({rel(Path(port_dir))}): `edit` changes only files there. "
                  "Other files in your stage directory are written with write_file.")


def _read_text_beneath(root: Path, parts, cap: int, what: str):
    try:
        dfd = open_dir_beneath(root, parts[:-1])
    except OSError as exc:
        return None, f"{what} cannot be opened ({exc.__class__.__name__}: a link or an unreadable directory)."
    try:
        data, why = read_file(Path(parts[-1]), cap, dir_fd=dfd)
    finally:
        os.close(dfd)
    if data is None:
        return None, f"{what} {why}."
    try:
        return data.decode("utf-8"), None
    except UnicodeDecodeError:
        return None, f"{what} is not UTF-8 text."


def _text_arg(stage: Path, text, name, flag: str):
    """(text, None) from `--old TEXT` or `--old-file FILE` (the file lies in the run directory, with no link on the way), or (None, reason)."""
    if text is not None and name is not None:
        return None, f"give {flag} or {flag}-file, not both."
    if name is None:
        return text, None
    run = run_dir_of(stage)
    if not name or "\0" in name or len(name) > 500:
        return None, f"{flag}-file: the file name is empty, too long or has a NUL."
    for base in (run, stage):
        cand = os.path.normpath(os.path.join(base, name))
        if _inside(run, cand):
            parts = Path(os.path.relpath(cand, run)).parts
            got, why = _read_text_beneath(run, parts, EDIT_TEXT_MAX, f"{flag}-file {clean(name, 100)}")
            if got is not None or "missing" not in str(why):
                return got, why
    return None, f"{flag}-file {clean(name, 100)} is missing or outside the run directory."


def _positions(text: str, old: str, limit: int) -> list:
    """Offsets of `old` in `text`, overlapping ones included, at most `limit` of them."""
    found, at = [], text.find(old)
    while at != -1 and len(found) < limit:
        found.append(at)
        at = text.find(old, at + 1)
    return found


def _closest_lines(text: str, old: str) -> list:
    """(line number, text) of the lines of `text` most like the first non-blank line of `old`, best first."""
    import difflib
    probe = next((l.strip() for l in old.splitlines() if l.strip()), "")[:200]
    if not probe:
        return []
    matcher = difflib.SequenceMatcher(None, autojunk=False)
    matcher.set_seq2(probe)
    scored, seen = [], set()
    for n, line in enumerate(text.splitlines(), start=1):
        body = line.strip()
        if not body or body in seen:
            continue
        matcher.set_seq1(body)
        if matcher.real_quick_ratio() < 0.5 or matcher.quick_ratio() < 0.5:
            continue
        ratio = matcher.ratio()
        if ratio >= 0.5:
            seen.add(body)
            scored.append((-ratio, n, body))
    scored.sort()
    return [(n, body) for _, n, body in scored[:EDIT_SHOWN]]


def _lines_changed(old: str, new: str) -> int:
    """The lines an edit changes: the larger of the old and new text's lines once the lines they share at the start and the end are set aside."""
    a, b = old.split("\n"), new.split("\n")
    while a and b and a[0] == b[0]:
        a, b = a[1:], b[1:]
    while a and b and a[-1] == b[-1]:
        a, b = a[:-1], b[:-1]
    return max(len(a), len(b))


def _stat_key(st):
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)


def edit_cmd(stage: Path, argv: list):
    """(exit code, lines) for `edit FILE --old TEXT --new TEXT` (or --old-file / --new-file): replace the one place where the old text occurs in a file
    of <stage>/port. Refused, with the file untouched: a file outside the port directory or reached through a link, an old text that occurs in no place or in
    several, a result that is empty or over the write bound, a file that changed after it was read. The new file is written to a temporary file in the
    same directory, flushed, and renamed over the old one, so a reader sees the old bytes or the new ones. Exit 0 changed, 1 no unique match, 2 refused."""
    try:
        args = _edit_parser().parse_args(argv)
    except _EditArgs as exc:
        return 2, [f"edit: {exc}. Usage: edit FILE --old TEXT --new TEXT (or --old-file / --new-file for text over several lines)."]
    old, why = _text_arg(stage, args.old, args.old_file, "--old")
    if old is None:
        return 2, [f"edit: {why if why else '--old (or --old-file) is needed.'}"]
    new, why = _text_arg(stage, args.new, args.new_file, "--new")
    if new is None:
        return 2, [f"edit: {why if why else '--new (or --new-file) is needed; an empty --new \"\" deletes the old text.'}"]
    if old == "":
        return 2, ["edit: the old text is empty, and it would match everywhere. Give the text to replace."]
    if old == new:
        return 2, ["edit: the old and the new text are the same, so nothing would change."]
    parts, why = port_target(stage, args.file)
    if parts is None:
        return 2, [f"edit: {why}"]
    shown = "/".join(rel(stage).split("/") + list(parts)) if rel(stage) != str(stage) else "/".join(parts)
    try:
        dfd = open_dir_beneath(stage, parts[:-1])
    except OSError as exc:
        return 2, [f"edit: {shown} cannot be reached ({exc.__class__.__name__}: a link on the way, or a directory that is not there). Nothing was changed."]
    tmp = None
    try:
        leaf = parts[-1]
        try:
            before = os.stat(leaf, dir_fd=dfd, follow_symlinks=False)
        except OSError as exc:
            return 2, [f"edit: {shown} cannot be read ({exc.__class__.__name__}). Nothing was changed."]
        if not stat.S_ISREG(before.st_mode):
            return 2, [f"edit: {shown} is not a regular file (a link, a directory, a pipe or a device). Nothing was changed."]
        if before.st_size > EDIT_MAX_BYTES:
            return 2, [f"edit: {shown} is {before.st_size:,} bytes, over the limit of {EDIT_MAX_BYTES:,} for a port file. Nothing was changed; write a shorter file."]
        data, why = read_file(Path(leaf), EDIT_MAX_BYTES, dir_fd=dfd)
        if data is None:
            return 2, [f"edit: {shown} {why}. Nothing was changed."]
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return 2, [f"edit: {shown} is not UTF-8 text. Nothing was changed."]
        at = _positions(text, old, EDIT_PLACES)
        if not at:
            near = _closest_lines(text, old)
            hint = ("The closest lines are " + "; ".join(f"line {n}: {clean(b, 100)!r}" for n, b in near) + ". ") if near else ""
            return 1, [f"edit: --old does not occur in {shown}. {hint}Copy the text exactly (indentation and line breaks included) from "
                       f"`show {shown} --lines A:B`, then run `edit` again. Nothing was changed."]
        if len(at) > 1:
            where = ", ".join(str(text.count("\n", 0, i) + 1) for i in at)
            count = f"{len(at)} places" if len(at) < EDIT_PLACES else f"at least {EDIT_PLACES} places"
            return 1, [f"edit: --old occurs in {count} of {shown} (lines {where}). Add more of the surrounding text to --old so that it matches one place only. "
                       "Nothing was changed."]
        result = text[:at[0]] + new + text[at[0] + len(old):]
        out = result.encode("utf-8")
        lines_after = result.count("\n") + 1
        if not result.strip():
            return 2, [f"edit: the result would be empty. Nothing was changed."]
        if len(out) > EDIT_MAX_BYTES or lines_after > EDIT_MAX_LINES:
            return 2, [f"edit: the result would be {len(out):,} bytes and {lines_after:,} lines; the limit for a port file is {EDIT_MAX_BYTES:,} bytes and "
                       f"{EDIT_MAX_LINES:,} lines. Nothing was changed; make the new text shorter."]
        try:
            tmp = f".edit-{os.getpid()}-{os.urandom(4).hex()}.tmp"
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0), 0o600, dir_fd=dfd)
            try:
                os.fchmod(fd, stat.S_IMODE(before.st_mode))
                view = memoryview(out)
                while view:
                    view = view[os.write(fd, view):]
                os.fsync(fd)
            finally:
                os.close(fd)
            if _stat_key(os.stat(leaf, dir_fd=dfd, follow_symlinks=False)) != _stat_key(before):
                raise OSError("the file changed after it was read")
            os.rename(tmp, leaf, src_dir_fd=dfd, dst_dir_fd=dfd)
            tmp = None
        except OSError as exc:
            return 2, [f"edit: {shown} was not changed: {clean(str(exc), 150)}. The file is unchanged; read it again with `show` and repeat the edit."]
        try:
            os.fsync(dfd)
        except OSError:
            pass
        record = {"tool": "edit", "file": shown, "bytes_before": len(data), "bytes_after": len(out), "lines_changed": _lines_changed(old, new),
                  "chars_old": len(old), "chars_new": len(new)}
        return 0, [f"edit: changed {record['lines_changed']} line(s) of {shown} ({len(data):,} to {len(out):,} bytes).", RECORD_MARK + json.dumps(record, sort_keys=True)]
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp, dir_fd=dfd)
            except OSError:
                pass
        os.close(dfd)



COMMANDS = {"last-check": "which check failed, where, the numbers and the evidence file",
            "layer-diff": "the port against the CPU reference layer by layer: the first layer whose relative error is over --threshold",
            "facts": "one value of port_facts.json by dotted path, or --find TEXT",
            "show": "a range of a file in the run directory with line numbers: show FILE --lines A:B",
            "cheat": "one entry of the verified TTNN call sheet (--sheet ttlang for the kernel sheet): cheat NAME",
            "edit": "change one exact piece of a port file: edit FILE --old TEXT --new TEXT (or --old-file / --new-file)"}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) == 5 and argv[0] == "--layer-diff-inner":          # the inner part, started by `layer-diff` under the reference interpreter
        return inner(argv[1:])
    if not argv or argv[0] not in COMMANDS:
        print("usage: python3 stage_tools.py <tool> [arguments]\n" + "\n".join(f"  {k}: {v}" for k, v in COMMANDS.items()))
        return 2
    if argv[0] == "last-check":
        print("\n".join(last_check(HERE)))
        return 0
    if argv[0] == "layer-diff":
        return layer_diff(HERE, argv[1:])
    code, lines = {"facts": facts_cmd, "show": show_cmd, "cheat": cheat_cmd, "edit": edit_cmd}[argv[0]](HERE, argv[1:])
    print("\n".join(lines))
    return code


if __name__ == "__main__":
    sys.exit(main())
