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

A second limit (issue 390): the cleanup at the end of a run finds the processes below this one, including orphans (this process is a
subreaper). If the supervisor kills the command's process group on a timeout or an abort, this file dies in the same kill and its cleanup
does not run. A helper the port started in a session of its own (`setsid`, a double fork) is not in that group. The supervisor now stops it:
every command it spawns carries ORCHARD_SPAWN_TAG in its environment, and after a kill it stops and names the processes that carry the tag
(orchard/agent.py, `spawn_checked`). A helper that replaces its environment (`env -i`, an exec with a new one) is not found. Catching that
needs kernel-level containment (a cgroup or a PID namespace), which this file cannot provide.

Port module interface (stages/2/port/port_model.py):

    load(model_dir: str, mesh_device) -> model                   required
        Load the weights from the Hugging Face snapshot directory onto the open device. The time this takes is the record's load_s.
    score_pairs(model, pairs: list[list[int]], media=None) -> list[float]    required for kind "reranker"
        `pairs` holds the token ids of whole prompts (prefix, query, document and suffix already in place). Return one number per
        prompt: logit("yes") - logit("no") at the prompt's last position. A list of floats or a torch or numpy array of that length.
    embed(model, ids: list[int], media=None) -> list[float]                  required for kind "embedding"
        Return the vector for one prompt: the last token's hidden state after the final norm, L2 normalised, as the Hugging Face
        reference computes it. The check takes the cosine, so the length does not matter, but the vector must have the reference's
        length, finite numbers, and some non-zero value. The check calls it once for each input of the pair file.
    close(model) -> None                                         required
        Release what load took. The device itself is closed by this file.
    COMPUTES_ON = "in-process" | "server"                        optional, default "in-process"
    server_address(model) -> "127.0.0.1:PORT"                    required when COMPUTES_ON is "server"
        A port that forwards the prompts to a model server, and does not run the model in this worker, says so (issue 407). The address
        is a loopback host (127.0.0.1, localhost or [::1]) and a port, called after `load` in a request of its own, so its time is not in load_s. The record then says the scores were
        computed by a server at that address, with the result of this file's own probe of /health and /v1/models; it does not say the
        chip this check opened ran them.

Images and video (issue 377). A row of the pair file may list `media`: files that belong to the prompt (an image or a video clip). The
port's two functions then take a second argument, `media`: for `score_pairs` a list with one entry for each prompt, for `embed` the entry
for that one prompt; each entry is a list of {"kind": "image" or "video", "name": the file's name, "sha256": hex, "data": bytes}. The
bytes are the file as stage 1 recorded it (this file reads it without following a link, checks its size and checks its sha256 against
the pair file's before the port sees it). The port decodes them itself; the ids already hold the placeholder tokens. When no row has media
the functions are called without the argument, so a text-only port needs no change.

The record's `computed_on` says where the scores were computed (issue 407). `chips` and `load_s` keep their meaning: `chips` are the
chips this check opened, and `load_s` is the time of the port's own `load` call. For a server-backed port neither says which chip ran
the model or how long the model took to load, and the record says so. A declared server that answers the probe shows that something
answers at the address, and does not show that it produced the scores.

Inputs: stages/1/evidence/reference/scoring-pairs.json and scoring-reference.json (orchard/scoring_reference.py). Output:
evidence/scoring-check.json (orchard-scoring-check/1, see orchard/scoring_check.py) and the last line of stdout. For kind "reranker"
the record holds the chip's scores; for kind "embedding" it holds the chip's vectors (the gate recomputes each cosine from them).

This file uses the standard library only, and reads files through `read_regular` below (it does not follow a link, does not wait on a
FIFO and stops at a size limit) because it runs alone in the run directory, where `orchard.safeio` is not importable.
"""
import hashlib
import http.client
import importlib
import json
import math
import os
import re
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
KINDS = ("reranker", "embedding")
RETRIEVAL_K = 10                 # choice: the bundles report NDCG@10; the gate's copy has the same value (a test checks the two agree)
MAX_DIM = 16384                  # choice: the length of one vector; the largest public embedding models are 8192
MAX_VECTOR_NUMEL = 500_000       # choice: numbers in all chip vectors of one record (a record this size is about 10 MB, under JSON_MAX)
MEDIA_DIR = "stages/1/evidence/reference/media/"      # media files named by the pair file live here and nowhere else
MEDIA_KINDS = ("image", "video")
MEDIA_MAX = 16 * 1024 * 1024     # choice: one media file; the gate reads it through a reader capped at JSON_MAX (16 MiB)
MEDIA_TOTAL_MAX = 128 * 1024 * 1024
MEDIA_PER_ROW = 8
MEDIA_FILES_MAX = 64
ORDER_GAP = 0.5                 # choice: the bundles count a pair of documents as ordered when the reference separates them by this much (logits)
MAX_PAIRS = 4096                 # choice: the fixed set has a few dozen pairs; more than this is not the set the recorder wrote
MAX_IDS = 1 << 20                # choice: ids in one prompt; the model card's limit is 8192
MAX_VALUE = 1e6                  # a score beyond this is not a logit difference
JSON_MAX = 16 * 1024 * 1024      # choice: the pair file is a few MB at most (32 prompts of up to 8192 ids)
REPLY_TIMEOUT_S = 3000.0         # the supervisor's own deadline is the limit that matters; this ends a worker that stopped answering
MIN_LOAD_S = 1e-6                # a load the clock cannot resolve is reported as this, not as 0
PREFIX = "port| "
COMPUTES_ON_VALUES = ("in-process", "server")
LOOPBACK_ADDRESS = re.compile(r"(127\.0\.0\.1|localhost|\[::1\]):([0-9]{1,5})")        # a server the port declares: loopback only
PROBE_TIMEOUT_S = 10.0
PROBE_REPLY_MAX = 1 << 20
MARK = "SCORING_CHECK"           # the start of the line the parent prints last; a port that prints it fails the run
FORGED = threading.Event()       # set when a line from the worker mentions MARK


def finite_score(x) -> bool:
    """True for an int or float that is finite and at most MAX_VALUE in size. An int of more than 64 bits is refused before any float
    conversion, because math.isfinite raises OverflowError on an int too large for a float (a 401-digit JSON integer)."""
    if type(x) not in (int, float) or (type(x) is int and x.bit_length() > 64):
        return False
    return math.isfinite(x) and abs(x) <= MAX_VALUE


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
    den = math.sqrt(saa) * math.sqrt(sbb)         # not sqrt(saa * sbb): the product of two small variances can underflow to 0
    return sab / den if den > 0.0 else 0.0


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


def cosine(a: list, b: list) -> float:
    """Cosine of two equal-length vectors. 0.0 when either is all zeros (the cosine is undefined). Clamped to [-1, 1]. Each vector is divided by
    its largest absolute value first, so a vector of tiny finite numbers (squares that would underflow to 0.0) has the same cosine as the same
    vector scaled up, and one whose largest value is 1 has a norm of at least 1."""
    ma = max((abs(x) for x in a), default=0.0)
    mb = max((abs(y) for y in b), default=0.0)
    if ma <= 0.0 or mb <= 0.0:
        return 0.0
    a = [x / ma for x in a]
    b = [y / mb for y in b]
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return max(-1.0, min(1.0, dot / na / nb))


def embed_numbers(reference: list, chip: list, kinds: list) -> dict:
    """The cosine of each chip vector to its reference vector: the mean and the minimum over all inputs and for each input kind. `kinds[i]`
    is the kind name of input i. Lists of equal, non-zero length."""
    cos = [cosine(r, c) for r, c in zip(reference, chip)]
    by_kind: dict = {}
    for k, v in zip(kinds, cos):
        by_kind.setdefault(k, []).append(v)
    return {"count": len(cos), "cosine_mean": sum(cos) / len(cos), "cosine_min": min(cos),
            "by_kind": {k: {"count": len(v), "mean": sum(v) / len(v), "min": min(v)} for k, v in by_kind.items()}}


def ndcg_at_k(scores: list, relevant: list, idx: list, k: int = RETRIEVAL_K):
    """NDCG@k of one group with binary relevance: the documents `idx` ranked by `scores` (a tie keeps the earlier document first). None when
    the group has no relevant document (the group is left out of the mean)."""
    order = sorted(idx, key=lambda i: (-scores[i], i))
    gains = sum(1.0 / math.log2(rank + 2) for rank, i in enumerate(order[:k]) if relevant[i])
    ideal = sum(1.0 / math.log2(rank + 2) for rank in range(min(k, sum(1 for i in idx if relevant[i]))))
    return gains / ideal if ideal > 0 else None


def retrieval(reference: list, chip: list, groups: list, relevant: list, k: int = RETRIEVAL_K):
    """NDCG@k and Hit@1 of the reference ranking and of the chip ranking against the pair file's relevance labels (issue 379), from the
    recorded scores only. None when no group has a relevant document."""
    members: dict = {}
    for i, g in enumerate(groups):
        members.setdefault(g, []).append(i)
    n_ref, n_chip, h_ref, h_chip, queries = [], [], 0, 0, 0
    for idx in members.values():
        a = ndcg_at_k(reference, relevant, idx, k)
        if a is None:
            continue
        queries += 1
        n_ref.append(a)
        n_chip.append(ndcg_at_k(chip, relevant, idx, k))
        h_ref += 1 if relevant[first_max(reference, idx)] else 0
        h_chip += 1 if relevant[first_max(chip, idx)] else 0
    if not queries:
        return None
    nr, nc = sum(n_ref) / queries, sum(n_chip) / queries
    return {"k": k, "queries": queries, "ndcg_reference": nr, "ndcg_chip": nc, "ndcg_delta": nc - nr,
            "hit1_reference": h_ref / queries, "hit1_chip": h_chip / queries, "hit1_delta": (h_chip - h_ref) / queries}


# ---- the worker: the port module's side of the pipes --------------------------------------------------------------------------

def as_floats(values) -> list:
    """The values the port returned (list, numpy or torch) as a plain list. They are not converted: the parent checks that each is a
    finite int or float, so a string or a bool is refused there and is not turned into a number here."""
    if hasattr(values, "tolist"):
        values = values.tolist()
    return list(values)


def read_media(path, max_bytes: int = MEDIA_MAX) -> bytes:
    """The bytes of the media file a pair row names: a plain relative path of one file directly under MEDIA_DIR, opened one component at a time
    beneath the run directory with O_NOFOLLOW on every component (a directory or the file replaced by a link, between one read and the next,
    is refused and nothing outside the run directory is read), no blocking, regular files only, under `max_bytes`. Raises OSError."""
    if not isinstance(path, str) or not path.startswith(MEDIA_DIR) or "\0" in path or ".." in Path(path).parts:
        raise OSError(f"media path {str(path)[:80]!r} is not a plain path under {MEDIA_DIR}")
    name = path[len(MEDIA_DIR):]
    if not name or "/" in name:
        raise OSError(f"media path {path} is not a file directly under {MEDIA_DIR}")
    cloexec = getattr(os, "O_CLOEXEC", 0)
    fd = os.open(RUN_DIR, os.O_RDONLY | os.O_DIRECTORY | cloexec)          # the run directory itself is the anchor
    try:
        for part in MEDIA_DIR.strip("/").split("/"):
            nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | cloexec, dir_fd=fd)
            os.close(fd)
            fd = nxt
        file_fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | cloexec, dir_fd=fd)
    finally:
        os.close(fd)
    try:
        if not stat.S_ISREG(os.fstat(file_fd).st_mode):
            raise OSError(f"{path} is not a regular file")
        chunks, total = [], 0
        while True:
            chunk = os.read(file_fd, 1 << 16)
            if not chunk:
                return b"".join(chunks)
            total += len(chunk)
            if total > max_bytes:
                raise OSError(f"{path} is larger than {max_bytes} bytes")
            chunks.append(chunk)
    finally:
        os.close(file_fd)


def load_media(entries: list) -> list:
    """The media of one prompt as the port receives it: the bytes of each file, read without following a link and under MEDIA_MAX, and
    checked against the sha256 the pair file names (a file changed after the parent hashed it is refused here, at the point of use)."""
    out = []
    for e in entries:
        data = read_media(e["path"])
        if sha_bytes(data) != e["sha256"]:
            raise OSError(f"media file {e['path']} does not match the sha256 in the pair file")
        out.append({"kind": e["kind"], "name": Path(e["path"]).name, "sha256": e["sha256"], "data": data})
    return out


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


def become_subreaper() -> None:
    """Make this process the parent of any orphaned descendant (Linux PR_SET_CHILD_SUBREAPER). A child of the port that exits leaves its own
    children running; without this they would belong to init and a scan below the worker could not find them."""
    import ctypes
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
        raise CheckFailed("this process cannot become a subreaper (prctl PR_SET_CHILD_SUBREAPER failed), so it cannot be sure no process "
                          "the port started is left running")


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
                reply({"ok": True, "computes_on": getattr(mod, "COMPUTES_ON", "in-process")})
            elif op == "load":           # timed: only the port's own load
                model = mod.load(req["model_dir"], device)
                reply({"ok": True})
            elif op == "address":        # untimed: a server-backed port names its server after the load timer has stopped
                reply({"ok": True, "server_address": str(mod.server_address(model))})
            elif op == "score":
                ids_list = [[int(i) for i in ids] for ids in req["pairs"]]
                if req.get("media") is None:
                    scores = mod.score_pairs(model, ids_list)
                else:
                    scores = mod.score_pairs(model, ids_list, media=[load_media(m) for m in req["media"]])
                reply({"ok": True, "scores": as_floats(scores)})
            elif op == "embed":          # one request holds a few inputs; the port is called once for each
                vectors = []
                for n, ids in enumerate(req["inputs"]):
                    ids = [int(i) for i in ids]
                    vec = mod.embed(model, ids) if req.get("media") is None else mod.embed(model, ids, media=load_media(req["media"][n]))
                    vectors.append(as_floats(vec))
                reply({"ok": True, "vectors": vectors})
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
        cmd = [sys.executable, "-u", "-B", "-P", "-s", str(Path(__file__).resolve()), "--worker", str(c_in), str(c_out)]
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

    def embed(self, label: str, inputs: list, dim: int, media=None) -> list:
        """The vectors for a few inputs: one list of `dim` finite numbers (not all zero) for each, or the run fails. `media` is None or one
        list of media entries for each input."""
        req = {"op": "embed", "inputs": inputs}
        if media is not None:
            req["media"] = media
        vectors = self.ask(**req).get("vectors")
        if not isinstance(vectors, list) or len(vectors) != len(inputs):
            raise CheckFailed(f"the port returned {len(vectors) if isinstance(vectors, list) else vectors!r} vectors for the {len(inputs)} inputs of {label}")
        for v in vectors:
            if not isinstance(v, list) or len(v) != dim:
                raise CheckFailed(f"the port returned a vector of length {len(v) if isinstance(v, list) else v!r} in {label}; the reference's vectors have length {dim}")
            if not all(finite_score(x) for x in v):
                raise CheckFailed(f"the port returned a vector with a value that is not a finite number in {label}")
            if not any(v):
                raise CheckFailed(f"the port returned a vector of all zeros in {label}: it has no direction")
        return [[float(x) for x in v] for v in vectors]

    def score(self, group: str, pairs: list, media=None) -> list:
        """The scores for one group of prompts: exactly one finite number per prompt, or the run fails."""
        req = {"op": "score", "pairs": pairs}
        if media is not None:
            req["media"] = media
        scores = self.ask(**req).get("scores")
        if not isinstance(scores, list) or len(scores) != len(pairs):
            raise CheckFailed(f"the port returned {len(scores) if isinstance(scores, list) else scores!r} scores for the {len(pairs)} prompts of group {group!r}")
        for s in scores:
            if not finite_score(s):
                raise CheckFailed(f"the port returned a score that is not a finite number in group {group!r}: {s!r}")
        return [float(s) for s in scores]

    def descendants(self) -> list:
        """The pids of every process below this one, found through /proc: the worker, what it started, and what its children left behind
        when they exited (this process is a subreaper, so an orphan is re-parented to it and not to init)."""
        parents = {}
        for entry in os.listdir("/proc"):
            if entry.isdigit():
                try:
                    parents[int(entry)] = int(read_regular(f"/proc/{entry}/stat", 1 << 16).rsplit(b")", 1)[1].split()[1])
                except (OSError, ValueError, IndexError):
                    continue
        found, frontier = [], [os.getpid()]
        while frontier:
            kids = [p for p, pp in parents.items() if pp in frontier and p not in found]
            found.extend(kids)
            frontier = kids
        return found

    def stop(self) -> None:
        """Make sure nothing the port started is running before the record is written. The worker is frozen first (SIGSTOP stops every
        thread in it, so no thread and no atexit callback can start a process or write a file during the scan), then every process
        below it is frozen and the scan is repeated until it finds no new one, then all of them and the worker are killed. The worker
        stays in this process's own process group on purpose: the supervisor kills that group when the test times out or the run is
        aborted, and SIGKILL skips this file's cleanup. The worker's output is read to the end."""
        try:
            os.kill(self.proc.pid, signal.SIGSTOP)
        except OSError:
            pass
        frozen: list = []
        for _ in range(20):                              # a frozen tree gains no members; the bound only guards a scan that never settles
            new = [p for p in self.descendants() if p not in frozen]
            if not new:
                break
            for pid in new:
                try:
                    os.kill(pid, signal.SIGSTOP)
                except OSError:
                    pass
            frozen += new
        for pid in reversed(frozen):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        self.proc.kill()
        self.proc.wait()
        try:
            self.wr.close()
        except OSError:
            pass
        for t in self.pumps:
            t.join(10)
        self.confirm_none_left()

    def confirm_none_left(self) -> None:
        """After the kill: reap what died (a killed orphan is this process's zombie child) and scan once more. A process still there after
        a few seconds means the port holds something this file cannot stop, so the run fails and no record is written."""
        for _ in range(100):
            try:
                while os.waitpid(-1, os.WNOHANG)[0]:
                    pass
            except ChildProcessError:
                pass
            if not self.descendants():
                return
            time.sleep(0.05)
        raise CheckFailed("a process the port started is still running after the check killed everything below it: no record is written")


def load_json(rel_or_path, what: str):
    path = rel_or_path if isinstance(rel_or_path, Path) else RUN_DIR / rel_or_path
    try:
        raw = read_regular(path)
        return json.loads(raw.decode("utf-8")), sha_bytes(raw)
    except (OSError, ValueError) as exc:
        raise CheckFailed(f"{what} cannot be read as JSON: {exc}")


def check_media(rows: list) -> list:
    """Check each row's `media` entries and hash the files. Returns, for each row, the list of entries the worker is sent (kind, path,
    sha256), or None for every row when no row has media. A kind other than image or video, a path outside MEDIA_DIR, a file that is a link,
    a FIFO or over MEDIA_MAX, a hash that differs from the pair file's, too many files or too many bytes in all each fail the run."""
    per_row, files, total, any_media = [], 0, 0, False
    for i, p in enumerate(rows):
        entries = p.get("media")
        if entries is None:
            per_row.append([])
            continue
        if not isinstance(entries, list) or len(entries) > MEDIA_PER_ROW:
            raise CheckFailed(f"{PAIRS} row {i} media must be a list of at most {MEDIA_PER_ROW} entries")
        sent = []
        for e in entries:
            if (not isinstance(e, dict) or e.get("kind") not in MEDIA_KINDS or not isinstance(e.get("path"), str)
                    or not isinstance(e.get("sha256"), str) or len(e["sha256"]) != 64):
                raise CheckFailed(f"{PAIRS} row {i} has a media entry that needs a kind ({'/'.join(MEDIA_KINDS)}), a path and a sha256")
            files += 1
            if files > MEDIA_FILES_MAX:
                raise CheckFailed(f"{PAIRS} lists more than {MEDIA_FILES_MAX} media files")
            try:
                data = read_media(e["path"])
            except OSError as exc:
                raise CheckFailed(f"{PAIRS} row {i}: the media file {str(e['path'])[:80]!r} cannot be read: {exc}")
            total += len(data)
            if total > MEDIA_TOTAL_MAX:
                raise CheckFailed(f"the media files of {PAIRS} are larger than {MEDIA_TOTAL_MAX} bytes in all")
            if sha_bytes(data) != e["sha256"]:
                raise CheckFailed(f"{PAIRS} row {i}: the media file {e['path']} does not match the sha256 the pair file names")
            sent.append({"kind": e["kind"], "path": e["path"], "sha256": e["sha256"]})
        any_media = any_media or bool(sent)
        per_row.append(sent)
    return per_row if any_media else None


def read_inputs(kind: str) -> dict:
    """The inputs of a run of `kind`, each checked for type and size: the pair rows, the reference (scores for a reranker, vectors for an
    embedding model), the sha256 of both files, and the media each row names. The reference names the pair file's sha256, and it must be
    the file on disk."""
    pairs_doc, pairs_sha = load_json(PAIRS, PAIRS)
    ref_doc, ref_sha = load_json(REFERENCE, REFERENCE)
    rows = pairs_doc.get("pairs") if isinstance(pairs_doc, dict) else None
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_PAIRS:
        raise CheckFailed(f"{PAIRS} holds no list of 1 to {MAX_PAIRS} pairs")
    name_key = "group" if kind == "reranker" else "kind"          # a reranker's rows are grouped by query; an embedding input has a kind
    for i, p in enumerate(rows):
        ids = p.get("input_ids") if isinstance(p, dict) else None
        if (not isinstance(p.get(name_key) if isinstance(p, dict) else None, str) or not isinstance(ids, list) or not 1 <= len(ids) <= MAX_IDS
                or not all(type(t) is int and t >= 0 for t in ids)):
            raise CheckFailed(f"{PAIRS} pair {i} needs a {name_key} name and a non-empty list of token ids")
        if kind == "reranker" and "relevant" in p and type(p["relevant"]) is not bool:
            raise CheckFailed(f"{PAIRS} pair {i} relevant must be true or false when it is present")
    named = ref_doc.get("pairs") if isinstance(ref_doc, dict) else None
    if not isinstance(named, dict) or named.get("sha256") != pairs_sha:
        raise CheckFailed(f"{REFERENCE} does not name {PAIRS} as it is on disk (its pair file changed after the reference was recorded)")
    if ref_doc.get("model_kind") != kind or pairs_doc.get("model_kind") != kind:
        raise CheckFailed(f"the reference files are for kind {ref_doc.get('model_kind')!r}, and this check runs kind {kind!r}")
    out = {"rows": rows, "pairs_sha": pairs_sha, "ref_sha": ref_sha, "ref_doc": ref_doc, "media": check_media(rows)}
    if kind == "reranker":
        scores = ref_doc.get("scores")
        if not isinstance(scores, list) or len(scores) != len(rows) or not all(finite_score(s) for s in scores):
            raise CheckFailed(f"{REFERENCE} holds no list of {len(rows)} finite scores")
        out["reference"] = [float(s) for s in scores]
        out["relevant"] = [p["relevant"] for p in rows] if all("relevant" in p for p in rows) else None
        return out
    vectors = ref_doc.get("vectors")
    if (not isinstance(vectors, list) or len(vectors) != len(rows) or not isinstance(vectors[0], list) or not 1 <= len(vectors[0]) <= MAX_DIM
            or len(vectors) * len(vectors[0]) > MAX_VECTOR_NUMEL):
        raise CheckFailed(f"{REFERENCE} holds no list of {len(rows)} vectors of one length (1 to {MAX_DIM}, at most {MAX_VECTOR_NUMEL} numbers in all)")
    for i, v in enumerate(vectors):
        if not isinstance(v, list) or len(v) != len(vectors[0]) or not all(finite_score(x) for x in v) or not any(v):
            raise CheckFailed(f"{REFERENCE} vector {i} is not a list of {len(vectors[0])} finite numbers with some value that is not zero")
    out["reference"] = [[float(x) for x in v] for v in vectors]
    return out


def probe_server(host: str, port: int) -> dict:
    """Ask the declared server /health and /v1/models without a proxy and without following a redirect. Returns {"health_status": int or None,
    "models": [ids], "error": text or None}. A server that does not answer is recorded, and the gate refuses the record."""
    out = {"health_status": None, "models": [], "error": None}
    for path in ("/health", "/v1/models"):
        conn = http.client.HTTPConnection(host, port, timeout=PROBE_TIMEOUT_S)
        try:
            conn.request("GET", path)
            resp = conn.getresponse()
            raw = resp.read(PROBE_REPLY_MAX + 1)
            if path == "/health":
                out["health_status"] = resp.status
            elif resp.status == 200 and len(raw) <= PROBE_REPLY_MAX:
                try:
                    data = json.loads(raw.decode("utf-8")).get("data")
                    out["models"] = [str(m["id"])[:200] for m in data[:8] if isinstance(m, dict) and "id" in m] if isinstance(data, list) else []
                except (ValueError, AttributeError, TypeError):
                    out["models"] = []
        except (OSError, http.client.HTTPException) as exc:
            out["error"] = f"{type(exc).__name__}: {exc}"[:300]
            break
        finally:
            conn.close()
    return out


def computed_on_record(where, address) -> dict:
    """The record's `computed_on` (issue 407) from what the port declared: where its scores are computed and, for a server, its address.
    Refuses a value that is not one of COMPUTES_ON_VALUES and a server address that is not a loopback host and a port."""
    if where not in COMPUTES_ON_VALUES:
        raise CheckFailed(f"the port's COMPUTES_ON is {where!r}; it must be one of {COMPUTES_ON_VALUES}")
    if where == "in-process":
        return {"where": "in-process", "server": None,
                "chips": "the chips this check opened; the port ran the model on them in the worker process",
                "load_s": "the time of the port's own load call, which loaded the model in the worker process"}
    m = LOOPBACK_ADDRESS.fullmatch(address) if isinstance(address, str) else None
    if m is None or not 1 <= int(m.group(2)) <= 65535:
        raise CheckFailed(f"the port declares a server at {address!r}; the address must be a loopback host and a port "
                          "(127.0.0.1:PORT, localhost:PORT or [::1]:PORT)")
    host = m.group(1)[1:-1] if m.group(1).startswith("[") else m.group(1)
    return {"where": "server", "server": {"address": address, "probe": probe_server(host, int(m.group(2)))},
            "chips": "the chips this check opened; the model ran in the server, which this record does not tie to them",
            "load_s": "the time of the port's own load call; the server's own model-load time is not measured separately"}


def port_sources() -> list:
    """Every .py file under the port directory with its sha256 as the worker saw it (the worker is stopped before this runs)."""
    out = []
    for f in sorted(PORT_DIR.rglob("*.py")):
        out.append({"path": str(f.relative_to(RUN_DIR)), "sha256": sha_file(f), "role": "model"})
    return out


UNHASHED_SUFFIXES = (".pyc", ".pyo", ".so", ".pyd", ".pth")


def refuse_links() -> None:
    """A link under the port directory (a file or a directory) could point at code that is not hashed, so any link fails the run. So does
    anything Python can import that the manifest does not hash: a `__pycache__` directory or a .pyc, .pyo, .so, .pyd or .pth file. An
    import would otherwise load cached bytecode that differs from the .py file the gate reads."""
    for root, dirs, files in os.walk(PORT_DIR):
        for name in dirs + files:
            if name == "__pycache__" or name.endswith(UNHASHED_SUFFIXES):
                raise CheckFailed(f"{os.path.relpath(os.path.join(root, name), RUN_DIR)} is importable code the check does not hash "
                                  "(bytecode, a compiled module or a path file): stages/2/port/ may hold .py files and plain data only")
            if os.path.islink(os.path.join(root, name)):
                raise CheckFailed(f"{os.path.relpath(os.path.join(root, name), RUN_DIR)} is a link: files under stages/2/port/ must be regular files")


EMBED_BATCH = 4                  # choice: inputs in one request; the reply for four vectors of 8192 numbers is well under a megabyte


def main() -> int:
    config, config_sha = load_json(CONFIG_FILE, CONFIG_FILE.name)
    model_dir = config.get("model_snapshot") if isinstance(config, dict) else None
    if not isinstance(model_dir, str) or not model_dir or "\0" in model_dir:
        raise CheckFailed(f"{CONFIG_FILE.name} names no model_snapshot")
    kind = config.get("kind")
    if kind not in KINDS:
        raise CheckFailed(f"{CONFIG_FILE.name} kind is {kind!r}; this check runs one of {KINDS}")
    chips = [c for c in os.environ.get("TT_VISIBLE_DEVICES", "").split(",") if c]
    if not chips:
        raise CheckFailed("TT_VISIBLE_DEVICES names no chip: this test runs only under the supervisor's lease")
    if not (PORT_DIR / f"{PORT_MODULE}.py").is_file():
        raise CheckFailed(f"stages/2/port/{PORT_MODULE}.py does not exist: the port module is the file the coder writes")
    refuse_links()
    inputs = read_inputs(kind)
    rows, reference, media = inputs["rows"], inputs["reference"], inputs["media"]
    pairs_sha, ref_sha, ref_doc = inputs["pairs_sha"], inputs["ref_sha"], inputs["ref_doc"]
    order: list = []                       # group names in order of first appearance (a reranker's requests go one group at a time)
    by_group: dict = {}
    if kind == "reranker":
        for i, p in enumerate(rows):
            if p["group"] not in by_group:
                order.append(p["group"])
            by_group.setdefault(p["group"], []).append(i)
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(RECORD):
        os.unlink(RECORD)

    become_subreaper()
    sources = port_sources()               # the files as they are before the worker starts; the record carries this manifest
    chip = [None] * len(rows)
    w = Worker()
    try:
        declared = w.ask(op="init", n_chips=len(chips)).get("computes_on")
        t0 = time.monotonic()
        w.ask(op="load", model_dir=model_dir)
        load_s = max(time.monotonic() - t0, MIN_LOAD_S)        # stops before the address is asked for: load_s is the port's load call
        computed_on = computed_on_record(declared, w.ask(op="address").get("server_address") if declared == "server" else None)     # full precision; a rounded value could be 0.0, which the gate refuses
        print(f"loaded in {load_s}s", flush=True)
        t1 = time.monotonic()
        if kind == "reranker":
            for g in order:
                idx = by_group[g]
                for i, s in zip(idx, w.score(g, [rows[i]["input_ids"] for i in idx], None if media is None else [media[i] for i in idx])):
                    chip[i] = s
        else:
            dim = len(reference[0])
            for start in range(0, len(rows), EMBED_BATCH):
                idx = list(range(start, min(start + EMBED_BATCH, len(rows))))
                for i, v in zip(idx, w.embed(f"inputs {idx[0]} to {idx[-1]}", [rows[i]["input_ids"] for i in idx], dim,
                                             None if media is None else [media[i] for i in idx])):
                    chip[i] = v
        score_s = max(time.monotonic() - t1, MIN_LOAD_S)
        w.ask(op="close")
    finally:
        w.stop()
    refuse_links()                  # again: a link made while the port ran
    if port_sources() != sources:
        raise CheckFailed("a file under stages/2/port/ changed, appeared or disappeared while the port ran: the code that produced the scores "
                          "must be the code that is recorded")
    if FORGED.is_set():
        raise CheckFailed(f"the port printed a line that mentions {MARK}: only this file prints that line, so the run is refused")

    common = {"schema": SCHEMA, "kind": kind,
              "reference": {"path": REFERENCE, "sha256": ref_sha, "precision": ref_doc.get("precision")},
              "pairs": {"path": PAIRS, "sha256": pairs_sha, "count": len(rows)}}
    tail = {"chips": chips, "load_s": load_s, "score_s": score_s, "computed_on": computed_on, "port_sources": sources, "template_sha256": sha_file(__file__),
            "config": {"path": str(CONFIG_FILE.relative_to(RUN_DIR)), "sha256": config_sha}}
    if kind == "reranker":
        groups = [p["group"] for p in rows]
        numbers = parity(reference, chip, groups)
        record = {**common, "groups": groups, "pair_kinds": [p.get("kind") for p in rows],
                  "reference_scores": reference, "chip_scores": chip, "metrics": numbers, **tail}
        if inputs["relevant"] is not None:
            record["retrieval"] = retrieval(reference, chip, groups, inputs["relevant"])
        summary = (f"pearson {numbers['pearson']:.5f}, mean |diff| {numbers['mean_abs_diff']:.4f}, max |diff| {numbers['max_abs_diff']:.4f}, "
                   f"top-1 {numbers['top1_groups_matching']}/{numbers['top1_groups']} groups, "
                   f"order kept {numbers['order_pairs_kept']}/{numbers['order_pairs']}")
    else:
        kinds = [p["kind"] for p in rows]
        numbers = embed_numbers(reference, chip, kinds)
        record = {**common, "input_kinds": kinds, "dim": len(reference[0]), "chip_vectors": chip, "metrics": numbers, **tail}
        summary = f"cosine mean {numbers['cosine_mean']:.5f}, min {numbers['cosine_min']:.5f} over {numbers['count']} inputs"
    data = (json.dumps(record, indent=1) + "\n").encode("utf-8")
    write_atomic(RECORD, data)
    print(summary, flush=True)
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
