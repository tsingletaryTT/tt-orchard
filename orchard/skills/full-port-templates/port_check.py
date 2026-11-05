# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 2 of a full port: the supervisor's own check of a port. The coder does not edit this file and does not run it.

The supervisor copies this file to stages/2/port_check.py, drafts stages/2/port_check_config.json next to it, and runs

    python3 -P -s stages/2/port_check.py

under the hardware lease. `-P` keeps the script's directory off sys.path and `-s` keeps the user site off it, so a file the coder
leaves in stages/2/ (a sitecustomize.py, a json.py) is not imported by this process. The gate (orchard/full_port.py) compares this
file's sha256 with the shipped one.

The coder's code is the port module stages/2/port/port_model.py (the interface is below). It never runs in this process. This process
(the parent) holds everything the record depends on: the reference ids, the comparison, the free-run text, the chip names, the timing
and the record. It starts a second process (the worker, this same file with `--worker`) that opens the device, imports the port module
and answers requests over two private pipes. The parent takes one thing from the worker per request: an integer token id. Nothing the
port module does inside the worker (printing a PORT_CHECK line, writing evidence/port-check.json, patching functions in this file,
installing an import hook) reaches the parent's code, its record or its output:

- the worker's stdout and stderr are read by the parent and printed with the prefix "port| ", so no line the port prints can begin
  with PORT_CHECK_SHA256, and a port that prints the word at all fails the run;
- the parent deletes any evidence/port-check.json before it starts the worker and writes the record itself after the worker has
  been stopped, through a temporary file and a rename;
- the last line the parent prints is `PORT_CHECK_SHA256 <sha256 of the record's bytes>`.

What this does not prove: the worker is the coder's code on the chip, and a port that returns the reference's ids without computing
them cannot be told from one that computed them. The gate's source checks (a port file must mention ttnn and must not read the stage 1
reference) make the plain forms visible. A person who wants the proof replays this command on a chip.

Port module interface (stages/2/port/port_model.py), all four functions required:

    load(model_dir: str, mesh_device) -> model
        Load the weights from the Hugging Face snapshot directory onto the open device. `mesh_device` is what
        ttnn.open_device (one chip) or ttnn.open_mesh_device (several) returned. The time this call takes is the record's load_s.
    prefill(model, ids: list[int]) -> logits_last
        Reset the model's cache, run the prompt, return the logits for the next token (a 1-D sequence of floats, or a torch or
        numpy array of any shape that flattens to the vocabulary).
    decode_step(model, token_id: int, position: int) -> logits
        Append `token_id` at `position` (the index of that token in the whole sequence, the prompt included), return the logits
        for the token after it.
    close(model) -> None
        Release what load took. The device itself is closed by this file.

Output: evidence/port-check.json (orchard-port-check/1, see orchard/skills/full-port-check.md) and the last line of stdout.
"""
import hashlib
import importlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

SCHEMA = "orchard-port-check/1"
HERE = Path(__file__).resolve().parent                    # stages/2
RUN_DIR = HERE.parents[1]
PORT_DIR = HERE / "port"
PORT_MODULE = "port_model"
CONFIG_FILE = HERE / "port_check_config.json"
RECORD = HERE / "evidence" / "port-check.json"
REFERENCE = "stages/1/evidence/reference/generated-ids.json"
PROMPT = "stages/1/evidence/reference/prompt-ids.json"
MAX_POSITIONS = 64               # choice: enough to give the agreement a spread, short enough to run in minutes on a cold cache
FREE_RUN_TOKENS = 48             # choice: a few sentences of text for the person who reads it
EXIT_GRACE_S = 60.0              # how long a worker that was asked to close may take to exit before it is killed
REPLY_TIMEOUT_S = 3000.0         # the supervisor's own deadline is the limit that matters; this ends a worker that stopped answering
MIN_LOAD_S = 1e-6                # a load the clock cannot resolve is reported as this, not as 0
PREFIX = "port| "
MARK = "PORT_CHECK"              # the start of the line the parent prints last; a port that prints it fails the run
FORGED = threading.Event()       # set when a line from the worker mentions MARK


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path) -> str:
    return sha_bytes(Path(path).read_bytes())


# ---- the worker: the port module's side of the pipes ------------------------------------------------------------------------

def argmax(logits) -> int:
    """The index of the largest value in whatever the port returned (list, numpy or torch)."""
    flat = logits
    if hasattr(logits, "reshape") and hasattr(logits, "argmax"):          # numpy and torch tensors
        return int(logits.reshape(-1).argmax())
    flat = list(logits)
    best, at = flat[0], 0
    for i, v in enumerate(flat):
        if v > best:
            best, at = v, i
    return int(at)


# An exception's text goes into a reply line that the supervisor's ledger, prompt and narration quote. URL user information and token-shaped words
# are taken out of it BEFORE it is cut to REPLY_CHARS: a cut between a password and its `@` would leave a password prefix that no pattern downstream
# can recognise (review of PR 432). The scheme's length is bounded and the rest of the match stops at the next `/`, so the time is linear in the message and a hostile
# one cannot make it quadratic; the supervisor scrubs the whole output again (orchard/port_repair.py `safe_text`).
REPLY_CHARS = 2000
_USERINFO = re.compile(r"([A-Za-z][A-Za-z0-9+.-]{0,31}://)[^/\s@]*@")
_TOKEN = re.compile(r"\b(?:hf_|ghp_|gho_|ghu_|ghs_|ghr_|github_pat_)[A-Za-z0-9_]{8,}")


def clip(text: str) -> str:
    """The first REPLY_CHARS characters of `text`, with URL user information and token-shaped words removed from the whole text first."""
    return _TOKEN.sub("[token removed]", _USERINFO.sub(r"\1", str(text)))[:REPLY_CHARS]


def open_device(ttnn, n_chips: int):
    """One chip: a device. Several: a mesh with the 1-D fabric on (collectives such as all_reduce need it, and the port module is imported
    after the mesh is open, so it cannot set it up). The closer turns the fabric off again, as scripts/minicpm3/run_chip.py does."""
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

    model = mod = device = closer = None
    for line in rd:
        try:
            req = json.loads(line)
            op = req["op"]
            if op == "init":             # untimed: import the runtime, open the device, import the port module
                try:
                    # The runtime is imported here, inside the guard, and before the port directory is on sys.path, so a file named ttnn.py
                    # there cannot stand in for it. A failure of the import (a missing dependency, a broken install) is the runtime's, like a
                    # failed device open: no edit to the port can fix it, so the parent calls it setup (issue 420, review of PR 425).
                    import ttnn
                    device, closer = open_device(ttnn, int(req["n_chips"]))
                except (Exception, SystemExit) as exc:
                    reply({"ok": False, "kind": "setup", "error": clip(f"{type(exc).__name__}: {exc}")})
                    continue
                sys.path.insert(0, str(PORT_DIR))
                mod = importlib.import_module(PORT_MODULE)
                reply({"ok": True})
            elif op == "load":           # timed: only the port's own load
                model = mod.load(req["model_dir"], device)
                reply({"ok": True})
            elif op == "prefill":
                reply({"ok": True, "id": argmax(mod.prefill(model, [int(i) for i in req["ids"]]))})
            elif op == "decode":
                reply({"ok": True, "id": argmax(mod.decode_step(model, int(req["token"]), int(req["position"])))})
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
            reply({"ok": False, "error": clip(f"{type(exc).__name__}: {exc}")})
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                return 1
    return 0


# ---- the parent --------------------------------------------------------------------------------------------------------------

class PortFailed(Exception):
    """The check failed. The message is printed and the run exits 2. `kind` is "port" when the port did not do what a request asked (or its
    files are wrong), which the coder can repair, and "setup" when the run around the port is wrong (no leased chip, a stage 1 file that does
    not read), which no edit to the port can fix. `op` names the request that failed, when one did. The supervisor reads both from the
    structured line printed after the message (orchard/port_repair.py) to decide whether the coder gets a repair round (issue 420)."""

    def __init__(self, message, kind="port", op=None):
        super().__init__(message)
        self.kind, self.op = kind, op


class Worker:
    """The worker process and its pipes. `ask` sends one request and returns the reply's integer id (or None)."""

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

    def ask(self, **req):
        self.wr.write(json.dumps(req) + "\n")
        self.wr.flush()
        box = []
        reader = threading.Thread(target=lambda: box.append(self.rd.readline()), daemon=True)
        reader.start()
        reader.join(REPLY_TIMEOUT_S)
        if not box or not box[0]:
            raise PortFailed(f"the port worker did not answer {req['op']!r} (it exited or stopped answering)", op=req["op"])
        try:
            answer = json.loads(box[0])
        except ValueError:
            raise PortFailed(f"the port worker's answer to {req['op']!r} was not JSON", op=req["op"])
        if not isinstance(answer, dict) or answer.get("ok") is not True:
            runtime = isinstance(answer, dict) and answer.get("kind") == "setup"       # only the worker's own device-open failure says so
            raise PortFailed(f"the port failed at {req['op']!r}: {answer.get('error') if isinstance(answer, dict) else answer!r}", op=req["op"],
                             kind="setup" if runtime else "port")
        if "id" in answer and (type(answer["id"]) is not int or answer["id"] < 0):
            raise PortFailed(f"the port returned an id that is not a whole number at {req['op']!r}: {answer['id']!r}", op=req["op"])
        return answer.get("id")

    def kill_descendants(self) -> None:
        """SIGKILL every process below the worker (children the port started), found through /proc before the worker exits and they are
        re-parented. The worker stays in this process's own process group on purpose: the supervisor kills that group when the test times
        out or the run is aborted, and SIGKILL skips this file's cleanup, so a worker in a group of its own would survive it. Processes that
        detach themselves with a new session or a double fork are not reached here. After a kill the supervisor stops them through the
        ORCHARD_SPAWN_TAG its spawned commands carry (orchard/agent.py, `spawn_checked`, issue 390)."""
        parents = {}
        for entry in os.listdir("/proc"):
            if entry.isdigit():
                try:
                    with open(f"/proc/{entry}/stat", "rb") as f:
                        parents[int(entry)] = int(f.read().rsplit(b")", 1)[1].split()[1])
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


def read_ids(rel: str, key: str) -> list[int]:
    data = json.loads((RUN_DIR / rel).read_text(encoding="utf-8"))
    ids = data.get(key) if isinstance(data, dict) else None
    if not isinstance(ids, list) or not ids or not all(type(i) is int for i in ids):
        raise PortFailed(f"{rel} holds no list of whole numbers under {key!r}", kind="setup")
    return ids


def free_run_text(model_dir: str, ids: list[int]) -> str:
    """The text of the free-run ids, decoded with the model's own tokenizer.json."""
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json")).decode(ids, skip_special_tokens=True)


def port_sources() -> list[dict]:
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
                raise PortFailed(f"{os.path.relpath(os.path.join(root, name), RUN_DIR)} is a link: files under stages/2/port/ must be regular files")


def main() -> int:
    config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    model_dir = config["model_snapshot"]
    chips = [c for c in os.environ.get("TT_VISIBLE_DEVICES", "").split(",") if c]
    if not chips:
        raise PortFailed("TT_VISIBLE_DEVICES names no chip: this test runs only under the supervisor's lease", kind="setup")
    if not (PORT_DIR / f"{PORT_MODULE}.py").is_file():
        raise PortFailed(f"stages/2/port/{PORT_MODULE}.py does not exist: the port module is the file the coder writes")
    refuse_links()
    reference, prompt = read_ids(REFERENCE, "generated_ids"), read_ids(PROMPT, "prompt_ids")
    n = min(len(reference), MAX_POSITIONS)
    ref = reference[:n]
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    if RECORD.exists() or RECORD.is_symlink():
        RECORD.unlink()

    w = Worker()
    try:
        w.ask(op="init", n_chips=len(chips))
        t0 = time.monotonic()
        w.ask(op="load", model_dir=model_dir)
        load_s = max(time.monotonic() - t0, MIN_LOAD_S)     # full precision; a rounded value could be 0.0, which the gate refuses
        print(f"loaded in {load_s}s", flush=True)
        # Teacher forced: at position i the port's argmax is taken, then the reference id i is fed in, never the port's own.
        chip = [w.ask(op="prefill", ids=prompt)]
        for i in range(1, n):
            chip.append(w.ask(op="decode", token=ref[i - 1], position=len(prompt) + i - 1))
        # Free run: a fresh prefill, then the port's own greedy ids.
        free = [w.ask(op="prefill", ids=prompt)]
        for k in range(1, FREE_RUN_TOKENS):
            free.append(w.ask(op="decode", token=free[-1], position=len(prompt) + k - 1))
        w.ask(op="close")
    finally:
        w.stop()
    refuse_links()                  # again: a link made while the port ran
    if FORGED.is_set():
        raise PortFailed(f"the port printed a line that mentions {MARK}: only this file prints that line, so the run is refused")

    record = {"schema": SCHEMA,
              "reference": {"path": REFERENCE, "sha256": sha_file(RUN_DIR / REFERENCE)},
              "prompt": {"path": PROMPT, "sha256": sha_file(RUN_DIR / PROMPT)},
              "reference_ids": ref, "chip_ids": chip,
              "top1_agreement": sum(1 for a, b in zip(ref, chip) if a == b) / n,
              "chips": chips, "load_s": load_s, "free_run_ids": free, "free_run_text": free_run_text(model_dir, free),
              "port_sources": port_sources(), "template_sha256": sha_file(__file__),
              "config": {"path": str(CONFIG_FILE.relative_to(RUN_DIR)), "sha256": sha_file(CONFIG_FILE)}}
    data = (json.dumps(record, indent=2) + "\n").encode("utf-8")
    tmp = RECORD.with_name(RECORD.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, RECORD)
    print(f"top-1 agreement {record['top1_agreement']:.4f} over {n} positions", flush=True)
    print("PORT_CHECK_SHA256", sha_bytes(data), flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        sys.exit(worker(int(sys.argv[2]), int(sys.argv[3])))
    try:
        sys.exit(main())
    except PortFailed as exc:
        print(f"FAILED: {exc}", flush=True)
        print("PORT_CHECK_FAILURE " + json.dumps({"kind": exc.kind, "op": exc.op}), flush=True)      # after the text: the supervisor reads this line
        sys.exit(2)
