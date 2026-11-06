# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 6 of a full port: the supervisor's own quality and benchmark run of a port. Nobody edits this file and nobody runs it by hand.

The supervisor copies this file to stages/6/port_quality.py, writes stages/6/port_quality_config.json next to it, copies the port that
passed the earlier stage into stages/6/port/, and runs

    python3 -P -s stages/6/port_quality.py

under the hardware lease. It is the sibling of port_check.py (stages 2 and 3) and keeps its structure: the port module runs in a second
process (the worker) that answers requests over two private pipes, and this process (the parent) holds the reference files, the clock,
the arithmetic and the record. The port module has the interface port_check.py documents (load, prefill, decode_step, close), plus one
optional module-level name, `DTYPE_POLICY`, a dict that maps a weight group of orchard/decode_floor.py to a dtype name. A port that
declares none is assumed to hold every weight in bf16, and the record says so.

What the run does, in this order, on one loaded port:

1. Teacher-forced scoring. The same positions as stage 2 (the stage 1 reference ids, fed in one by one). At each position the worker
   returns the port's argmax, the natural log of the probability the port gives the reference's next id, and the port's logits at the
   five ids that stage 1 recorded as the reference's top five at that position (stages/1/evidence/reference/top5-logits.json). The
   perplexity and the logit-gap comparison are computed from these by the parent and by the gate.
2. A free run. A fresh prefill, then the port's own greedy ids for FREE_RUN_TOKENS tokens, decoded to text with the model's tokenizer.
3. A benchmark. WARMUP_RUNS discarded runs and REPEATS measured runs in the same process. Each run is one prefill of the prompt and
   BENCH_STEPS decode steps that feed the port its own greedy ids. The parent times each request round trip with its own clock, so the
   port cannot report a time. The round trip includes the pipe and the argmax, which is small next to a chip step; the record says so.

Output: evidence/port-quality.json (orchard-port-quality/1) and the last line of stdout, PORT_QUALITY_SHA256 <hex>.

What this does not show: a port that returns the reference's ids without computing them cannot be told from one that computed them. The
gate's source checks (a port file must mention ttnn and must not read the stage 1 reference) make the plain forms visible. Timing and
quality are measured on one prompt. The benchmark is batch size one.
"""
import errno
import hashlib
import importlib
import json
import math
import os
import re
import select
import signal
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

SCHEMA = "orchard-port-quality/1"
HERE = Path(__file__).resolve().parent                    # stages/6
RUN_DIR = HERE.parents[1]
PORT_DIR = HERE / "port"
PORT_MODULE = "port_model"
CONFIG_FILE = HERE / "port_quality_config.json"
RECORD = HERE / "evidence" / "port-quality.json"
REFERENCE = "stages/1/evidence/reference/generated-ids.json"
PROMPT = "stages/1/evidence/reference/prompt-ids.json"
TOP5 = "stages/1/evidence/reference/top5-logits.json"
MAX_POSITIONS = 64               # choice: the same as stage 2's teacher-forced run
FREE_RUN_TOKENS = 64             # choice: twice the reference's 32 ids, so the text has room to wander and the leading match has room to end
WARMUP_RUNS = 1                  # docs/benchmark-rules.md (orchard/resident.py WARMUP_RUNS)
REPEATS = 3                      # docs/benchmark-rules.md (orchard/resident.py MIN_REPEATS)
BENCH_STEPS = 32                 # decode steps in one benchmark run, after one prefill of the prompt
PROBES = 5                       # the reference's top five ids at each position
EXIT_GRACE_S = 60.0
REPLY_TIMEOUT_S = 3000.0
MIN_TIME_S = 1e-9                # a time the clock cannot resolve is reported as this, not as 0
PREFIX = "port| "
MARK = "PORT_QUALITY"            # the start of the line the parent prints last; a port that prints it fails the run
FORGED = threading.Event()       # set when a line from the worker mentions MARK


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


MAX_HASHED_BYTES = 256 * 1024 * 1024      # choice: far above any file this run hashes (the stage 1 reference files and the model's config.json)


def sha_file(path) -> str:
    """The sha256 of the regular file at `path`. A FIFO or device is refused with OSError at once: the port runs between the two hashes of
    a stage 1 file and could replace it with a FIFO, and a plain read would then block and hold the hardware lease until the deadline
    (review of PR 491). The file is opened nonblocking, checked on the opened descriptor, and read in chunks up to a cap. Symlinks are not
    refused here because the model's config.json is a link inside a Hugging Face snapshot; `refuse_links` covers the run directory."""
    fd = os.open(str(path), os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)  # hardening: ok no O_NOFOLLOW on purpose, the model's config.json is a snapshot link; the descriptor is checked regular
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError(errno.EINVAL, "not a regular file", str(path))
        digest, total = hashlib.sha256(), 0
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                return digest.hexdigest()
            total += len(chunk)
            if total > MAX_HASHED_BYTES:
                raise OSError(errno.EFBIG, "file too large to hash", str(path))
            digest.update(chunk)
    finally:
        os.close(fd)


def sha_or_none(path):
    try:
        return sha_file(path)
    except OSError:
        return None


# ---- the worker: the port module's side of the pipes ------------------------------------------------------------------------

def floats(logits) -> list:
    """The logits as a flat list of Python floats, from a list, a numpy array or a torch tensor (bf16 included)."""
    if hasattr(logits, "reshape") and hasattr(logits, "tolist"):
        flat = logits.reshape(-1)
        if hasattr(flat, "float"):                         # a torch tensor: bf16 does not convert to numpy directly
            flat = flat.float()
        return [float(v) for v in flat.tolist()]
    return [float(v) for v in logits]


def argmax(values: list) -> int:
    best, at = values[0], 0
    for i, v in enumerate(values):
        if v > best:
            best, at = v, i
    return at


def native_argmax(logits) -> int:
    """The index of the largest value, taken by the tensor or array itself when it has an argmax (as port_check.py does), else in Python."""
    if hasattr(logits, "reshape") and hasattr(logits, "argmax"):
        return int(logits.reshape(-1).argmax())
    return argmax(floats(logits))


def log_prob(values: list, target: int) -> float:
    """ln softmax(values)[target], with the maximum taken out first so nothing overflows."""
    if not 0 <= target < len(values):
        raise ValueError(f"target id {target} is outside the {len(values)} logits")
    top = max(values)
    # The target is shifted by the maximum before the log of the sum is taken off. Adding `top` back first loses the sum's logarithm when the
    # logits share a large offset (five logits of 1e20 would give 0.0 instead of -ln 5; review of PR 491).
    return (values[target] - top) - math.log(sum(math.exp(v - top) for v in values))


REPLY_CHARS = 2000
_USERINFO = re.compile(r"([A-Za-z][A-Za-z0-9+.-]{0,31}://)[^/\s@]*@")
_TOKEN = re.compile(r"\b(?:hf_|ghp_|gho_|ghu_|ghs_|ghr_|github_pat_)[A-Za-z0-9_]{8,}")


def clip(text: str) -> str:
    """The first REPLY_CHARS characters of `text`, with URL user information and token-shaped words removed first (as port_check.py does)."""
    return _TOKEN.sub("[token removed]", _USERINFO.sub(r"\1", str(text)))[:REPLY_CHARS]


def open_device(ttnn, n_chips: int):
    """One chip: a device. Several: a mesh with the 1-D fabric on (as port_check.py opens it, so a port that passed stage 3 loads the same way)."""
    if n_chips > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        dev = ttnn.open_mesh_device(ttnn.MeshShape(1, n_chips))

        def close_mesh(d):
            ttnn.close_mesh_device(d)
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        return dev, close_mesh
    return ttnn.open_device(device_id=0), ttnn.close_device


def declared_policy(mod):
    """The port's `DTYPE_POLICY` when it is a small dict of text to text, else None."""
    p = getattr(mod, "DTYPE_POLICY", None)
    if isinstance(p, dict) and 0 < len(p) <= 16 and all(isinstance(k, str) and isinstance(v, str) for k, v in p.items()):
        return dict(p)
    return None


def answer(logits, req) -> dict:
    """The reply for one prefill or decode request: the argmax, and when the request names a target and probe ids, the log probability of the
    target and the logits at the probes."""
    if "target" not in req:
        return {"ok": True, "id": native_argmax(logits)}      # only the id is needed: the whole vocabulary is not turned into a Python list
    v = floats(logits)
    out = {"ok": True, "id": argmax(v)}
    if "target" in req:
        out["logprob"] = log_prob(v, int(req["target"]))
        out["probe"] = [v[int(i)] for i in req.get("probe", [])]
    return out


def worker(read_fd: int, write_fd: int) -> int:
    """Answer requests until `close`. Every reply is one JSON line: {"ok": true, ...} or {"ok": false, "error": text}."""
    rd, wr = os.fdopen(read_fd, "r", encoding="utf-8"), os.fdopen(write_fd, "w", encoding="utf-8")  # hardening: ok private pipe descriptors this file's parent made

    def reply(obj):
        wr.write(json.dumps(obj) + "\n")
        wr.flush()

    model = mod = device = closer = None
    for line in rd:
        try:
            req = json.loads(line)
            op = req["op"]
            if op == "init":
                try:
                    import ttnn
                    device, closer = open_device(ttnn, int(req["n_chips"]))
                except (Exception, SystemExit) as exc:
                    reply({"ok": False, "kind": "setup", "error": clip(f"{type(exc).__name__}: {exc}")})
                    continue
                sys.path.insert(0, str(PORT_DIR))
                mod = importlib.import_module(PORT_MODULE)
                reply({"ok": True, "dtype_policy": declared_policy(mod)})
            elif op == "load":
                model = mod.load(req["model_dir"], device)
                reply({"ok": True})
            elif op == "prefill":
                reply(answer(mod.prefill(model, [int(i) for i in req["ids"]]), req))
            elif op == "decode":
                reply(answer(mod.decode_step(model, int(req["token"]), int(req["position"])), req))
            elif op == "close":
                if mod is not None and model is not None:
                    mod.close(model)
                if closer is not None and device is not None:
                    closer(device)
                model = device = None
                reply({"ok": True})
            else:
                reply({"ok": False, "error": f"unknown op {op!r}"})
        except BaseException as exc:
            reply({"ok": False, "error": clip(f"{type(exc).__name__}: {exc}")})
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                return 1
    return 0


# ---- the parent --------------------------------------------------------------------------------------------------------------

class QualityFailed(Exception):
    """The run failed. The message is printed and the run exits 2. `kind` is "port" when the port did not do what a request asked, and
    "setup" when the run around the port is wrong (no leased chip, a reference file that does not read)."""

    def __init__(self, message, kind="port", op=None):
        super().__init__(message)
        self.kind, self.op = kind, op


def proc_stat(pid: int):
    """(parent pid, start time in clock ticks) of a process from /proc/<pid>/stat, or None when it is gone or unreadable. The start time with the
    pid names one process for the life of the machine's boot."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:  # hardening: ok reads /proc/<pid>/stat, a kernel file
            rest = f.read().rsplit(b")", 1)[1].split()
        return int(rest[1]), int(rest[19])
    except (OSError, ValueError, IndexError):
        return None


def _pidfd_open(pid: int) -> int:
    """A pidfd for `pid`. Some Python builds lack os.pidfd_open, so the system call (434 on x86-64 and arm64) is the fallback; OSError when the
    kernel has no pidfds or the process is gone."""
    if hasattr(os, "pidfd_open"):
        return os.pidfd_open(pid)
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.syscall(434, pid, 0)
    if fd < 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    return fd


def _pidfd_kill(fd: int) -> None:
    """SIGKILL the process a pidfd names (pidfd_send_signal, call 424, when the signal module lacks it)."""
    if hasattr(signal, "pidfd_send_signal"):
        signal.pidfd_send_signal(fd, signal.SIGKILL)
        return
    import ctypes
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.syscall(424, fd, int(signal.SIGKILL), None, 0) < 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))


def kill_if_same(pid: int, started: int) -> bool:
    """SIGKILL `pid` only if it is still the process that started at `started`. A pidfd is opened first and the start time is read after that, so
    the signal goes to the process the descriptor names even if the number is reused a moment later. On a kernel without pidfds the start time
    is checked just before os.kill, which leaves a window of a few instructions."""
    try:
        fd = _pidfd_open(pid)
    except OSError as exc:
        if exc.errno in (errno.ENOSYS, errno.EPERM):  # no pidfds here: the start-time check alone
            now = proc_stat(pid)
            if now is None or now[1] != started:
                return False
            try:
                os.kill(pid, signal.SIGKILL)
                return True
            except OSError:
                return False
        return False
    try:
        now = proc_stat(pid)
        if now is None or now[1] != started:
            return False
        _pidfd_kill(fd)
        return True
    except OSError:
        return False
    finally:
        os.close(fd)


class Worker:
    """The worker process and its pipes. `ask` sends one request and returns the reply (a dict)."""

    def __init__(self):
        p_in, c_out = os.pipe()
        c_in, p_out = os.pipe()
        cmd = [sys.executable, "-u", "-P", "-s", str(Path(__file__).resolve()), "--worker", str(c_in), str(c_out)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,  # hardening: ok starts the worker; not a file open
                                     pass_fds=(c_in, c_out), cwd=str(RUN_DIR))
        os.close(c_in)
        os.close(c_out)
        self.rd_fd, self._buf = p_in, b""                                 # replies are read from the descriptor itself, with a deadline (`_readline`)
        self.wr = os.fdopen(p_out, "w", encoding="utf-8")  # hardening: ok private pipe descriptor this file's parent made
        self.pumps = [threading.Thread(target=self._pump, args=(s,), daemon=True) for s in (self.proc.stdout, self.proc.stderr)]
        for t in self.pumps:
            t.start()

    @staticmethod
    def _pump(stream) -> None:
        for raw in iter(stream.readline, b""):
            if MARK.encode() in raw:
                FORGED.set()
            sys.stdout.write(PREFIX + raw.decode("utf-8", "replace").rstrip("\r\n") + "\n")
            sys.stdout.flush()

    def _readline(self, timeout_s: float) -> str | None:
        """One reply line from the worker's pipe: the line, "" when the worker closed the pipe, None when `timeout_s` passed with no whole line.
        It waits on the descriptor with `select` and a deadline, so a request costs no thread of its own and a request that gets no reply leaves
        nothing blocked behind (review of PR 491)."""
        deadline = time.monotonic() + timeout_s
        while b"\n" not in self._buf:
            left = deadline - time.monotonic()
            if left <= 0:
                return None
            ready, _, _ = select.select([self.rd_fd], [], [], left)
            if not ready:
                return None
            chunk = os.read(self.rd_fd, 1 << 16)
            if not chunk:
                return ""
            self._buf += chunk
        line, _, self._buf = self._buf.partition(b"\n")
        return line.decode("utf-8", "replace")

    def ask(self, **req) -> dict:
        self.wr.write(json.dumps(req) + "\n")
        self.wr.flush()
        reply = self._readline(REPLY_TIMEOUT_S)
        if not reply:
            raise QualityFailed(f"the port worker did not answer {req['op']!r} (it exited or stopped answering)", op=req["op"])
        try:
            answer_ = json.loads(reply)
        except ValueError:
            raise QualityFailed(f"the port worker's answer to {req['op']!r} was not JSON", op=req["op"])
        if not isinstance(answer_, dict) or answer_.get("ok") is not True:
            runtime = isinstance(answer_, dict) and answer_.get("kind") == "setup"
            raise QualityFailed(f"the port failed at {req['op']!r}: {answer_.get('error') if isinstance(answer_, dict) else answer_!r}",
                                op=req["op"], kind="setup" if runtime else "port")
        if "id" in answer_ and (type(answer_["id"]) is not int or answer_["id"] < 0):
            raise QualityFailed(f"the port returned an id that is not a whole number at {req['op']!r}: {answer_['id']!r}", op=req["op"])
        if "logprob" in answer_:
            lp, probe = answer_["logprob"], answer_.get("probe")
            ok = (isinstance(lp, (int, float)) and not isinstance(lp, bool) and math.isfinite(lp) and lp <= 1e-9
                  and isinstance(probe, list) and len(probe) == len(req.get("probe", []))
                  and all(isinstance(p, (int, float)) and not isinstance(p, bool) and math.isfinite(p) for p in probe))
            if not ok:
                raise QualityFailed(f"the port's logits at {req['op']!r} give a log probability or probe value that is not a finite number "
                                    "(or the log probability is above zero)", op=req["op"])
        return answer_

    def kill_descendants(self) -> None:
        """SIGKILL every process below the worker (children the port started), found through /proc (the same method as port_check.py). The scan
        keeps each child's start time with its pid, and a child is signalled only if a pidfd taken afterwards still shows that start time, so a pid
        that was freed and reused between the scan and the signal is left alone (review of PR 491)."""
        parents = {}
        for entry in os.listdir("/proc"):
            if entry.isdigit():
                stat = proc_stat(int(entry))
                if stat is not None:
                    parents[int(entry)] = stat
        found, frontier = [], [self.proc.pid]
        while frontier:
            kids = [p for p, (pp, _) in parents.items() if pp in frontier and p not in found]
            found.extend(kids)
            frontier = kids
        for pid in reversed(found):
            kill_if_same(pid, parents[pid][1])

    def stop(self) -> None:
        self.kill_descendants()
        try:
            self.wr.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=EXIT_GRACE_S)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        for t in self.pumps:
            t.join(10)


def read_json(rel: str):
    return json.loads((RUN_DIR / rel).read_text(encoding="utf-8"))  # hardening: ok reads stage 1 files the gate hashes and compares through safeio


def read_ids(rel: str, key: str) -> list:
    data = read_json(rel)
    ids = data.get(key) if isinstance(data, dict) else None
    if not isinstance(ids, list) or not ids or not all(type(i) is int for i in ids):
        raise QualityFailed(f"{rel} holds no list of whole numbers under {key!r}", kind="setup")
    return ids


def reference_top5(n: int):
    """The reference's top five at each of the first `n` positions: [{"ids": [...], "logits": [...]}], or None when stage 1 wrote no usable file.
    Position k is the prediction of generated id k: stage 1's step k holds the top five of the logits after the prompt and k generated ids."""
    try:
        steps = read_json(TOP5).get("steps")
        out = []
        for k in range(n):
            top = steps[k]["top5"]
            ids = [int(t["id"]) for t in top[:PROBES]]
            logits = [float(t["logit"]) for t in top[:PROBES]]
            if len(ids) != PROBES or not all(math.isfinite(x) for x in logits):
                return None
            out.append({"ids": ids, "logits": logits})
        return out
    except (OSError, ValueError, KeyError, IndexError, TypeError, AttributeError):
        return None


def free_run_text(model_dir: str, ids: list) -> str:
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json")).decode(ids, skip_special_tokens=True)


def port_sources() -> list:
    out = []
    for f in sorted(PORT_DIR.rglob("*.py")):
        out.append({"path": str(f.relative_to(RUN_DIR)), "sha256": sha_file(f), "role": "model"})
    return out


def refuse_links() -> None:
    for root, dirs, files in os.walk(PORT_DIR):
        for name in dirs + files:
            if os.path.islink(os.path.join(root, name)):
                raise QualityFailed(f"{os.path.relpath(os.path.join(root, name), RUN_DIR)} is a link: files under stages/6/port/ must be regular files")


def loadavg_1m():
    try:
        return os.getloadavg()[0]
    except OSError:
        return None


def bench_run(w: Worker, prompt: list, warmup: bool) -> dict:
    """One benchmark run: a prefill, then BENCH_STEPS decode steps that feed the port its own ids. Timed by this process's clock."""
    t0 = time.monotonic()
    nxt = w.ask(op="prefill", ids=prompt)["id"]
    t1 = time.monotonic()
    for k in range(BENCH_STEPS):
        nxt = w.ask(op="decode", token=nxt, position=len(prompt) + k)["id"]
    t2 = time.monotonic()
    prefill_s, decode_s = max(t1 - t0, MIN_TIME_S), max(t2 - t1, MIN_TIME_S)
    return {"warmup": warmup, "prefill_s": prefill_s, "decode_s": decode_s, "steps": BENCH_STEPS,
            "decode_tok_s": BENCH_STEPS / decode_s, "host_loadavg_1m": loadavg_1m()}


def perplexity_text(nll: float) -> str:
    """exp(nll) to four places, or a word when it is too large for a float. The line this is printed on comes before the record's mark line, and an
    overflow here would end the run without the mark (review of PR 491, round 5); the supervisor reports the unrepresentable value as TODO."""
    try:
        return f"{math.exp(nll):.4f}"
    except OverflowError:
        return "too large to represent"


def main() -> int:
    config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))  # hardening: ok reads the config the supervisor wrote (its sha256 is in the record)
    model_dir = config["model_snapshot"]
    chips = [c for c in os.environ.get("TT_VISIBLE_DEVICES", "").split(",") if c]
    if not chips:
        raise QualityFailed("TT_VISIBLE_DEVICES names no chip: this test runs only under the supervisor's lease", kind="setup")
    if not (PORT_DIR / f"{PORT_MODULE}.py").is_file():
        raise QualityFailed(f"stages/6/port/{PORT_MODULE}.py does not exist: the supervisor copies the port that passed the earlier stage there")
    refuse_links()
    # Each stage 1 file is hashed before the port starts, and the record carries that hash: a port that rewrites one of them while it runs
    # cannot choose the file the gate later hashes (review of PR 491).
    stage1_sha = {rel: sha_or_none(RUN_DIR / rel) for rel in (REFERENCE, PROMPT, TOP5)}      # None for a missing file; its own check names it below
    reference, prompt = read_ids(REFERENCE, "generated_ids"), read_ids(PROMPT, "prompt_ids")
    n = min(len(reference), MAX_POSITIONS)
    ref = reference[:n]
    top5 = reference_top5(n)
    if top5 is None:
        raise QualityFailed(f"{TOP5} is missing or holds no top five for the first {n} positions: the comparison with the reference "
                            "has nothing to compare with", kind="setup")
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    if RECORD.exists() or RECORD.is_symlink():
        RECORD.unlink()
    # The model's config.json is hashed before the port runs. The floor and the share are worked out from that file, so a port that rewrites it
    # during load() or close() must not be able to choose the file the gate later hashes (review of PR 491).
    config_sha = sha_file(Path(model_dir) / "config.json")

    w = Worker()
    try:
        policy = w.ask(op="init", n_chips=len(chips)).get("dtype_policy")
        t0 = time.monotonic()
        w.ask(op="load", model_dir=model_dir)
        load_s = max(time.monotonic() - t0, MIN_TIME_S)
        print(f"loaded in {load_s}s", flush=True)
        # 1. Teacher forced: the port's argmax, the log probability of the reference id, and its logits at the reference's top five.
        chip, logprobs, probes = [], [], []
        for i in range(n):
            score = {"target": ref[i], "probe": top5[i]["ids"]}
            r = (w.ask(op="prefill", ids=prompt, **score) if i == 0
                 else w.ask(op="decode", token=ref[i - 1], position=len(prompt) + i - 1, **score))
            chip.append(r["id"])
            logprobs.append(r["logprob"])
            probes.append(r["probe"])
        # 2. Free run: a fresh prefill, then the port's own greedy ids.
        free = [w.ask(op="prefill", ids=prompt)["id"]]
        for k in range(1, FREE_RUN_TOKENS):
            free.append(w.ask(op="decode", token=free[-1], position=len(prompt) + k - 1)["id"])
        # 3. Benchmark: discarded warm-up runs, then the measured runs, in this one process.
        runs = [bench_run(w, prompt, True) for _ in range(WARMUP_RUNS)] + [bench_run(w, prompt, False) for _ in range(REPEATS)]
        w.ask(op="close")
    finally:
        w.stop()
    refuse_links()
    if sha_file(Path(model_dir) / "config.json") != config_sha:
        raise QualityFailed("the model's config.json changed while the port ran: the floor and the share would describe another configuration")
    for rel, before in stage1_sha.items():
        if sha_or_none(RUN_DIR / rel) != before:
            raise QualityFailed(f"{rel} changed while the port ran: the comparison would describe a file the port chose")
    if FORGED.is_set():
        raise QualityFailed(f"the port printed a line that mentions {MARK}: only this file prints that line, so the run is refused")

    record = {"schema": SCHEMA,
              "reference": {"path": REFERENCE, "sha256": stage1_sha[REFERENCE]},
              "prompt": {"path": PROMPT, "sha256": stage1_sha[PROMPT]},
              "top5": {"path": TOP5, "sha256": stage1_sha[TOP5]},
              "reference_ids": ref, "reference_top5": top5, "chip_ids": chip, "logprobs": logprobs, "probe_logits": probes,
              "chips": chips, "load_s": load_s, "free_run_ids": free, "free_run_text": free_run_text(model_dir, free),
              "bench": {"warmup_runs": WARMUP_RUNS, "repeats": REPEATS, "steps": BENCH_STEPS, "runs": runs,
                        "timed_by": "this process's monotonic clock around each request round trip (pipe and argmax included)"},
              "dtype_policy": policy if policy else {"default": "bf16"},
              "dtype_policy_source": "declared by the port (DTYPE_POLICY)" if policy else "assumed: the port declares no DTYPE_POLICY, so every weight is taken as bf16",
              "model_config": {"path": "config.json", "sha256": config_sha},
              "port_sources": port_sources(), "template_sha256": sha_file(__file__),
              "config": {"path": str(CONFIG_FILE.relative_to(RUN_DIR)), "sha256": sha_file(CONFIG_FILE)}}
    data = (json.dumps(record, indent=2) + "\n").encode("utf-8")
    tmp = RECORD.with_name(RECORD.name + ".tmp")
    tmp.write_bytes(data)  # hardening: ok writes the record that the gate re-reads through safeio and ties to the output mark
    os.replace(tmp, RECORD)
    measured = [r for r in runs if not r["warmup"]]
    print(f"perplexity {perplexity_text(-sum(logprobs) / n)} over {n} positions; "
          f"decode {sorted(r['decode_tok_s'] for r in measured)[len(measured) // 2]:.2f} tok/s (median of {len(measured)})", flush=True)
    print("PORT_QUALITY_SHA256", sha_bytes(data), flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        sys.exit(worker(int(sys.argv[2]), int(sys.argv[3])))
    try:
        sys.exit(main())
    except QualityFailed as exc:
        print(f"FAILED: {exc}", flush=True)
        print("PORT_QUALITY_FAILURE " + json.dumps({"kind": exc.kind, "op": exc.op}), flush=True)
        sys.exit(2)
