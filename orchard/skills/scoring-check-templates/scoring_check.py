# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 2 for a scoring model (issue 358): the supervisor's own parity check of a port of a reranker. The coder does not edit this file
and does not run it. It follows the full-port check (skills/full-port-templates/port_check.py) and differs in what it asks the port.

The supervisor copies this file to stages/2/scoring_check.py, drafts stages/2/scoring_check_config.json next to it, and runs

    python3 -P -s stages/2/scoring_check.py

under the hardware lease. `-P` keeps the script's directory off sys.path and `-s` keeps the user site off it. The gate
(orchard/scoring_check.py) compares this file's sha256 with the shipped one.

The coder's code is the port module stages/2/port/port_model.py. It never runs in this process. This process (the parent) holds the pair
set, the reference scores, the comparison and the record. It starts a second process (the worker, this same file with `--worker`) that
opens the device, imports the port module and answers requests over two private pipes. The parent takes one thing from the worker per
request: a list of finite numbers, one per pair it sent. Nothing the port does inside the worker (printing a SCORING_CHECK line, writing
evidence/scoring-check.json, patching functions in this file, installing an import hook) reaches the parent's code, its record or its
output:

- the worker's stdout and stderr are read by the parent and printed with the prefix "port| ", and a port that prints the word
  SCORING_CHECK at all fails the run;
- the parent deletes any evidence/scoring-check.json before it starts the worker and writes the record itself after the worker has been
  stopped, through a temporary file and a rename;
- the last line the parent prints is `SCORING_CHECK_SHA256 <sha256 of the record's bytes>`.

The port is given token ids and nothing else (no text, no reference score). It does not see the reference through this file. A port file
that reads the reference from the run directory is refused by the gate's source check.

What this does not prove: the worker is the coder's code on the chip, and a port that returns the reference's scores without computing
them cannot be told from one that computed them. The gate's source checks make the plain forms visible. A person who wants the proof
replays this command on a chip.

Port module interface (stages/2/port/port_model.py):

    load(model_dir: str, mesh_device) -> model                   required
        Load the weights from the Hugging Face snapshot directory onto the open device. The time this takes is the record's load_s.
    score_pairs(model, pairs: list[list[int]]) -> list[float]    required for kind "reranker"
        `pairs` holds the token ids of whole prompts (prefix, query, document and suffix already in place). Return one number per
        prompt: logit("yes") - logit("no") at the prompt's last position. A list of floats or a torch or numpy array of that length.
    embed(model, ids: list[int]) -> list[float]                  reserved for kind "embedding"; not called in this slice
        Return the unit-length vector for one prompt. The check refuses kind "embedding" until a later slice builds it.
    close(model) -> None                                         required
        Release what load took. The device itself is closed by this file.

Inputs: stages/1/evidence/reference/scoring-pairs.json and scoring-reference.json (orchard/scoring_reference.py). Output:
evidence/scoring-check.json (orchard-scoring-check/1, see orchard/scoring_check.py) and the last line of stdout.

This file uses the standard library only, and reads files through `read_regular` below (it does not follow a link, does not wait on a
FIFO and stops at a size limit) because it runs alone in the run directory, where `orchard.safeio` is not importable.
"""
import hashlib
import importlib
import json
import math
import os
import signal
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

SCHEMA = "orchard-scoring-check/1"
HERE = Path(__file__).resolve().parent                    # stages/2
RUN_DIR = HERE.parents[1]
PORT_DIR = HERE / "port"
PORT_MODULE = "port_model"
CONFIG_FILE = HERE / "scoring_check_config.json"
RECORD = HERE / "evidence" / "scoring-check.json"
PAIRS = "stages/1/evidence/reference/scoring-pairs.json"
REFERENCE = "stages/1/evidence/reference/scoring-reference.json"
KIND = "reranker"                # the only kind this slice runs
ORDER_GAP = 0.5                  # choice: the bundles count a pair of documents as ordered when the reference separates them by this much (logits)
MAX_PAIRS = 4096                 # choice: the fixed set has a few dozen pairs; more than this is not the set the recorder wrote
MAX_IDS = 1 << 20                # choice: ids in one prompt; the model card's limit is 8192
MAX_VALUE = 1e6                  # a score beyond this is not a logit difference
JSON_MAX = 16 * 1024 * 1024      # choice: the pair file is a few MB at most (32 prompts of up to 8192 ids)
EXIT_GRACE_S = 60.0
REPLY_TIMEOUT_S = 3000.0         # the supervisor's own deadline is the limit that matters; this ends a worker that stopped answering
MIN_LOAD_S = 1e-6                # a load the clock cannot resolve is reported as this, not as 0
PREFIX = "port| "
MARK = "SCORING_CHECK"           # the start of the line the parent prints last; a port that prints it fails the run
FORGED = threading.Event()       # set when a line from the worker mentions MARK


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_regular(path, max_bytes: int = JSON_MAX) -> bytes:
    """The bytes of a regular file. A link, a FIFO, a device, a directory or a file over `max_bytes` raises OSError. The file is opened
    without following a link and without blocking, and its type is checked on the open descriptor."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0))
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError(f"{path} is not a regular file")
        chunks, total = [], 0
        while True:
            chunk = os.read(fd, 1 << 16)
            if not chunk:
                return b"".join(chunks)
            total += len(chunk)
            if total > max_bytes:
                raise OSError(f"{path} is larger than {max_bytes} bytes")
            chunks.append(chunk)
    finally:
        os.close(fd)


def sha_file(path) -> str:
    return sha_bytes(read_regular(path))


def write_atomic(path: Path, data: bytes) -> None:
    """Write `data` to a temporary file next to `path` and rename it over `path`. The temporary name is created new (an existing name,
    or a link at that name, is an error)."""
    tmp = path.with_name(path.name + ".tmp")
    if os.path.lexists(tmp):
        os.unlink(tmp)
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0), 0o600)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)


# ---- the numbers: the parent computes them from the scores it was given -------------------------------------------------------
# orchard/scoring_check.py holds a second copy of `parity` (the gate recomputes). tests/test_scoring_check_template.py checks that the
# two give the same numbers on the same input.

def pearson(a: list, b: list) -> float:
    """Pearson r of two equal-length lists. 0.0 when either list is constant (the correlation is undefined, and a constant score is not
    a match)."""
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    sab = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    saa = sum((x - ma) ** 2 for x in a)
    sbb = sum((y - mb) ** 2 for y in b)
    if saa <= 0.0 or sbb <= 0.0:
        return 0.0
    return sab / math.sqrt(saa * sbb)


def first_max(values: list, idx: list) -> int:
    """The position in `idx` of the first largest value (a tie goes to the earlier pair)."""
    best = idx[0]
    for i in idx:
        if values[i] > values[best]:
            best = i
    return best


def parity(reference: list, chip: list, groups: list) -> dict:
    """The parity numbers of `chip` scores against `reference` scores. `groups[i]` is the group name of pair i. Lists of equal, non-zero
    length. Returns pearson, mean and max absolute difference, how many groups pick the same top document, and the share of pairs of
    documents within a group that keep the reference's order when the reference separates them by at least ORDER_GAP."""
    n = len(reference)
    diffs = [abs(a - b) for a, b in zip(reference, chip)]
    members: dict = {}
    for i, g in enumerate(groups):
        members.setdefault(g, []).append(i)
    top1 = sum(1 for idx in members.values() if first_max(reference, idx) == first_max(chip, idx))
    kept = total = 0
    for idx in members.values():
        for x in range(len(idx)):
            for y in range(x + 1, len(idx)):
                i, j = idx[x], idx[y]
                gap = reference[i] - reference[j]
                if abs(gap) >= ORDER_GAP:
                    total += 1
                    if (chip[i] - chip[j]) * gap > 0:
                        kept += 1
    return {"pearson": pearson(reference, chip), "mean_abs_diff": sum(diffs) / n, "max_abs_diff": max(diffs),
            "top1_groups_matching": top1, "top1_groups": len(members),
            "order_pairs": total, "order_pairs_kept": kept, "order_kept": (kept / total) if total else None}


# ---- the worker: the port module's side of the pipes --------------------------------------------------------------------------

def as_floats(values) -> list:
    """The values the port returned (list, numpy or torch) as a plain list. They are not converted: the parent checks that each is a
    finite int or float, so a string or a bool is refused there and is not turned into a number here."""
    if hasattr(values, "tolist"):
        values = values.tolist()
    return list(values)


def open_device(ttnn, n_chips: int):
    """One chip: a device. Several: a mesh with the 1-D fabric on, and a closer that turns it off again (the same as the port check)."""
    if n_chips > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        dev = ttnn.open_mesh_device(ttnn.MeshShape(1, n_chips))

        def close_mesh(d):
            ttnn.close_mesh_device(d)
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        return dev, close_mesh
    return ttnn.open_device(device_id=0), ttnn.close_device


def worker(read_fd: int, write_fd: int) -> int:
    """Answer requests until `close`. Every reply is one JSON line: {"ok": true, ...} or {"ok": false, "error": text}."""
    rd, wr = os.fdopen(read_fd, "r", encoding="utf-8"), os.fdopen(write_fd, "w", encoding="utf-8")

    def reply(obj):
        wr.write(json.dumps(obj) + "\n")
        wr.flush()

    import ttnn      # before the port directory is on sys.path, so a file named ttnn.py there cannot stand in for the runtime
    sys.path.insert(0, str(PORT_DIR))
    model = mod = device = closer = None
    for line in rd:
        try:
            req = json.loads(line)
            op = req["op"]
            if op == "init":             # untimed: open the device and import the port module
                device, closer = open_device(ttnn, int(req["n_chips"]))
                mod = importlib.import_module(PORT_MODULE)
                reply({"ok": True})
            elif op == "load":           # timed: only the port's own load
                model = mod.load(req["model_dir"], device)
                reply({"ok": True})
            elif op == "score":
                reply({"ok": True, "scores": as_floats(mod.score_pairs(model, [[int(i) for i in ids] for ids in req["pairs"]]))})
            elif op == "close":
                if mod is not None and model is not None:
                    mod.close(model)
                if closer is not None and device is not None:
                    closer(device)
                model = device = None
                reply({"ok": True})
                # Stay up until the parent closes the pipe: the parent looks for the port's child processes while this one is alive.
            else:
                reply({"ok": False, "error": f"unknown op {op!r}"})
        except BaseException as exc:             # the port's failure goes back as text; the parent decides what it means
            reply({"ok": False, "error": f"{type(exc).__name__}: {exc}"[:2000]})
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                return 1
    return 0


# ---- the parent ---------------------------------------------------------------------------------------------------------------

class CheckFailed(Exception):
    """The port or an input did not do what the check needs. The message is printed and the run exits non-zero."""


class Worker:
    """The worker process and its pipes. `ask` sends one request and returns the reply."""

    def __init__(self):
        p_in, c_out = os.pipe()          # worker -> parent replies
        c_in, p_out = os.pipe()          # parent -> worker requests
        cmd = [sys.executable, "-u", "-P", "-s", str(Path(__file__).resolve()), "--worker", str(c_in), str(c_out)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     pass_fds=(c_in, c_out), cwd=str(RUN_DIR))
        os.close(c_in)
        os.close(c_out)
        self.rd, self.wr = os.fdopen(p_in, "r", encoding="utf-8"), os.fdopen(p_out, "w", encoding="utf-8")
        self.pumps = [threading.Thread(target=self._pump, args=(s,), daemon=True) for s in (self.proc.stdout, self.proc.stderr)]
        for t in self.pumps:
            t.start()

    @staticmethod
    def _pump(stream) -> None:
        """Print each line the worker wrote with the prefix, so none can pass for a line of this file's."""
        for raw in iter(stream.readline, b""):
            if MARK.encode() in raw:
                FORGED.set()
            sys.stdout.write(PREFIX + raw.decode("utf-8", "replace").rstrip("\r\n") + "\n")
            sys.stdout.flush()

    def ask(self, **req) -> dict:
        try:
            self.wr.write(json.dumps(req) + "\n")
            self.wr.flush()
        except (OSError, ValueError):
            raise CheckFailed(f"the port worker is gone: it did not take {req['op']!r}")
        box = []
        reader = threading.Thread(target=lambda: box.append(self.rd.readline()), daemon=True)
        reader.start()
        reader.join(REPLY_TIMEOUT_S)
        if not box or not box[0]:
            raise CheckFailed(f"the port worker did not answer {req['op']!r} (it exited or stopped answering)")
        try:
            answer = json.loads(box[0])
        except ValueError:
            raise CheckFailed(f"the port worker's answer to {req['op']!r} was not JSON")
        if not isinstance(answer, dict) or answer.get("ok") is not True:
            raise CheckFailed(f"the port failed at {req['op']!r}: {answer.get('error') if isinstance(answer, dict) else answer!r}")
        return answer

    def score(self, group: str, pairs: list) -> list:
        """The scores for one group of prompts: exactly one finite number per prompt, or the run fails."""
        scores = self.ask(op="score", pairs=pairs).get("scores")
        if not isinstance(scores, list) or len(scores) != len(pairs):
            raise CheckFailed(f"the port returned {len(scores) if isinstance(scores, list) else scores!r} scores for the {len(pairs)} prompts of group {group!r}")
        for s in scores:
            if type(s) not in (int, float) or not math.isfinite(s) or abs(s) > MAX_VALUE:
                raise CheckFailed(f"the port returned a score that is not a finite number in group {group!r}: {s!r}")
        return [float(s) for s in scores]

    def kill_descendants(self) -> None:
        """SIGKILL every process below the worker (children the port started), found through /proc before the worker exits and they are
        re-parented. The worker stays in this process's own process group on purpose: the supervisor kills that group when the test times
        out or the run is aborted, and SIGKILL skips this file's cleanup."""
        parents = {}
        for entry in os.listdir("/proc"):
            if entry.isdigit():
                try:
                    parents[int(entry)] = int(read_regular(f"/proc/{entry}/stat", 1 << 16).rsplit(b")", 1)[1].split()[1])
                except (OSError, ValueError, IndexError):
                    continue
        found, frontier = [], [self.proc.pid]
        while frontier:
            kids = [p for p, pp in parents.items() if pp in frontier and p not in found]
            found.extend(kids)
            frontier = kids
        for pid in reversed(found):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass

    def stop(self) -> None:
        """Make sure the worker is gone before the record is written, so nothing it runs can write after. Its output is read to the end."""
        self.kill_descendants()
        try:
            self.wr.close()                              # a worker still waiting for a request sees the end of its input and exits
        except OSError:
            pass
        try:
            self.proc.wait(timeout=EXIT_GRACE_S)         # after `close` it exits on its own and flushes what it printed
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        for t in self.pumps:
            t.join(10)


def load_json(rel_or_path, what: str):
    path = rel_or_path if isinstance(rel_or_path, Path) else RUN_DIR / rel_or_path
    try:
        raw = read_regular(path)
        return json.loads(raw.decode("utf-8")), sha_bytes(raw)
    except (OSError, ValueError) as exc:
        raise CheckFailed(f"{what} cannot be read as JSON: {exc}")


def read_inputs() -> tuple:
    """(pairs, reference scores, pairs sha256, reference sha256), each checked for type and size. The reference names the pair file's
    sha256, and it must be the file on disk."""
    pairs_doc, pairs_sha = load_json(PAIRS, PAIRS)
    ref_doc, ref_sha = load_json(REFERENCE, REFERENCE)
    rows = pairs_doc.get("pairs") if isinstance(pairs_doc, dict) else None
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_PAIRS:
        raise CheckFailed(f"{PAIRS} holds no list of 1 to {MAX_PAIRS} pairs")
    for i, p in enumerate(rows):
        ids = p.get("input_ids") if isinstance(p, dict) else None
        if (not isinstance(p.get("group") if isinstance(p, dict) else None, str) or not isinstance(ids, list) or not 1 <= len(ids) <= MAX_IDS
                or not all(type(t) is int and t >= 0 for t in ids)):
            raise CheckFailed(f"{PAIRS} pair {i} needs a group name and a non-empty list of token ids")
    scores = ref_doc.get("scores") if isinstance(ref_doc, dict) else None
    if (not isinstance(scores, list) or len(scores) != len(rows)
            or not all(type(s) in (int, float) and math.isfinite(s) and abs(s) <= MAX_VALUE for s in scores)):
        raise CheckFailed(f"{REFERENCE} holds no list of {len(rows)} finite scores")
    named = ref_doc.get("pairs")
    if not isinstance(named, dict) or named.get("sha256") != pairs_sha:
        raise CheckFailed(f"{REFERENCE} does not name {PAIRS} as it is on disk (its pair file changed after the scores were recorded)")
    if ref_doc.get("model_kind") != KIND or pairs_doc.get("model_kind") != KIND:
        raise CheckFailed(f"the reference files are for kind {ref_doc.get('model_kind')!r}, and this check runs kind {KIND!r}")
    return rows, [float(s) for s in scores], pairs_sha, ref_sha, ref_doc


def port_sources() -> list:
    """Every .py file under the port directory with its sha256 as the worker saw it (the worker is stopped before this runs)."""
    out = []
    for f in sorted(PORT_DIR.rglob("*.py")):
        out.append({"path": str(f.relative_to(RUN_DIR)), "sha256": sha_file(f), "role": "model"})
    return out


def refuse_links() -> None:
    """A link under the port directory (a file or a directory) could point at code that is not hashed, so any link fails the run."""
    for root, dirs, files in os.walk(PORT_DIR):
        for name in dirs + files:
            if os.path.islink(os.path.join(root, name)):
                raise CheckFailed(f"{os.path.relpath(os.path.join(root, name), RUN_DIR)} is a link: files under stages/2/port/ must be regular files")


def main() -> int:
    config, config_sha = load_json(CONFIG_FILE, CONFIG_FILE.name)
    model_dir = config.get("model_snapshot") if isinstance(config, dict) else None
    if not isinstance(model_dir, str) or not model_dir or "\0" in model_dir:
        raise CheckFailed(f"{CONFIG_FILE.name} names no model_snapshot")
    if config.get("kind") != KIND:
        raise CheckFailed(f"{CONFIG_FILE.name} kind is {config.get('kind')!r}; this slice of the check runs {KIND!r} only "
                          "(embedding models are a later slice)")
    chips = [c for c in os.environ.get("TT_VISIBLE_DEVICES", "").split(",") if c]
    if not chips:
        raise CheckFailed("TT_VISIBLE_DEVICES names no chip: this test runs only under the supervisor's lease")
    if not (PORT_DIR / f"{PORT_MODULE}.py").is_file():
        raise CheckFailed(f"stages/2/port/{PORT_MODULE}.py does not exist: the port module is the file the coder writes")
    refuse_links()
    rows, reference, pairs_sha, ref_sha, ref_doc = read_inputs()
    order: list = []                       # group names in order of first appearance
    by_group: dict = {}
    for i, p in enumerate(rows):
        if p["group"] not in by_group:
            order.append(p["group"])
        by_group.setdefault(p["group"], []).append(i)
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(RECORD):
        os.unlink(RECORD)

    chip = [None] * len(rows)
    w = Worker()
    try:
        w.ask(op="init", n_chips=len(chips))
        t0 = time.monotonic()
        w.ask(op="load", model_dir=model_dir)
        load_s = max(time.monotonic() - t0, MIN_LOAD_S)     # full precision; a rounded value could be 0.0, which the gate refuses
        print(f"loaded in {load_s}s", flush=True)
        t1 = time.monotonic()
        for g in order:
            idx = by_group[g]
            for i, s in zip(idx, w.score(g, [rows[i]["input_ids"] for i in idx])):
                chip[i] = s
        score_s = max(time.monotonic() - t1, MIN_LOAD_S)
        w.ask(op="close")
    finally:
        w.stop()
    refuse_links()                  # again: a link made while the port ran
    if FORGED.is_set():
        raise CheckFailed(f"the port printed a line that mentions {MARK}: only this file prints that line, so the run is refused")

    groups = [p["group"] for p in rows]
    numbers = parity(reference, chip, groups)
    record = {"schema": SCHEMA, "kind": KIND,
              "reference": {"path": REFERENCE, "sha256": ref_sha, "precision": ref_doc.get("precision")},
              "pairs": {"path": PAIRS, "sha256": pairs_sha, "count": len(rows)},
              "groups": groups, "pair_kinds": [p.get("kind") for p in rows],
              "reference_scores": reference, "chip_scores": chip, "metrics": numbers,
              "chips": chips, "load_s": load_s, "score_s": score_s,
              "port_sources": port_sources(), "template_sha256": sha_file(__file__),
              "config": {"path": str(CONFIG_FILE.relative_to(RUN_DIR)), "sha256": config_sha}}
    data = (json.dumps(record, indent=2) + "\n").encode("utf-8")
    write_atomic(RECORD, data)
    print(f"pearson {numbers['pearson']:.5f}, mean |diff| {numbers['mean_abs_diff']:.4f}, max |diff| {numbers['max_abs_diff']:.4f}, "
          f"top-1 {numbers['top1_groups_matching']}/{numbers['top1_groups']} groups, "
          f"order kept {numbers['order_pairs_kept']}/{numbers['order_pairs']}", flush=True)
    print("SCORING_CHECK_SHA256", sha_bytes(data), flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        sys.exit(worker(int(sys.argv[2]), int(sys.argv[3])))
    try:
        sys.exit(main())
    except CheckFailed as exc:
        print(f"FAILED: {exc}", flush=True)
        sys.exit(2)
