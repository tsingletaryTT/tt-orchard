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
them cannot be told from one that computed them. The gate's source checks (a port file must make a reachable ttnn matmul-family call and
must not read the stage 1 reference) make the plain forms visible. A person who wants the proof replays this command on a chip.

Device-op gate (issue 541). In run 18 of the tiny Qwen4 rehearsal all 22 versions of the port made zero `ttnn.<op>(` calls: `import ttnn  # noqa`
and host torch. The "chip run" computed on the CPU. This file now counts, and fails the run with a message that says what to do, when
  * no source file under the port directory has a reachable matmul-family ttnn call (the static early reject, before the device is touched);
  * the first prefill makes fewer matmul-family ttnn calls than the floor, one per decoder layer of the checkpoint's config (counted at run
    time: the worker puts a counting proxy in `sys.modules["ttnn"]` before it imports the port module, so a mention, a comment, an unused
    import or a function nobody calls adds nothing);
  * no decode step makes that many (a step that replays a captured ttnn trace counts as the captured pass);
  * after `load`, the port's model holds more weight elements as host torch or numpy tensors than as ttnn tensors.
The counts go into the record as `device_ops` and the gate (orchard/full_port.py) checks them again. Limits: the count is of calls made, so a
port that calls ttnn.linear on a dummy tensor and computes elsewhere passes the count; the weight scan sees only what `load` returned and the
module's globals; and the proxy sees attribute access on `ttnn` and `from ttnn import name`, and misses `from ttnn.sub import name`.

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

Optional hook, for localising a numerics failure (issue 505):

    layer_outputs(model, ids: list[int]) -> list
        Run the prompt `ids` once and return the hidden state after each decoder layer, in order, as torch tensors or numpy arrays (any shape).
        It is called only when the check ends below the gate's agreement bar, in the same process and under the same lease, after the
        comparison is done. Its result never changes the record, the exit code or the gate. The worker saves the arrays to
        evidence/layer-port.npz and this file prints one line, `PORT_CHECK_LAYERS <sha256 of that file> <layer count>`, which the
        supervisor's capture of the output keeps, so `stage_tools.py layer-diff` can tell this file from one written afterwards.

Output: evidence/port-check.json (orchard-port-check/1, see orchard/skills/full-port-check.md) and the last line of stdout.
"""
import collections
import hashlib
import heapq
import importlib
import inspect
import itertools
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

SCHEMA = "orchard-port-check/1"
HERE = Path(__file__).resolve().parent                    # stages/2, stages/3, or stage 4's entry directory stages/4/configs/<N>
RUN_DIR = next(p for p in HERE.parents if p.name == "stages").parent      # the run directory: the nearest `stages` above this file's directory
PORT_DIR = HERE / "port"
PORT_MODULE = "port_model"
CONFIG_FILE = HERE / "port_check_config.json"
RECORD = HERE / "evidence" / "port-check.json"
LAYER_FILE = HERE / "evidence" / "layer-port.npz"         # written by the worker from the optional `layer_outputs` hook; advice, never evidence the gate reads
LAYER_MARK = "PORT_CHECK_LAYERS"                          # the parent's line that names that file (it starts with MARK, so the worker cannot print it)
LAYER_DIAG_BELOW = 0.85          # the hook runs only when the agreement is below this (orchard/full_port.py TOP1_MIN; a test keeps them equal)
MAX_LAYERS = 512                 # choice: more layers than any decoder stack; a longer list is refused
MAX_NUMEL = 2 ** 28              # choice: total elements saved (1 GiB as float32); a larger result is refused before anything is written
REFERENCE = "stages/1/evidence/reference/generated-ids.json"
PROMPT = "stages/1/evidence/reference/prompt-ids.json"
MAX_POSITIONS = 64               # choice: enough to give the agreement a spread, short enough to run in minutes on a cold cache
FREE_RUN_TOKENS = 48             # choice: a few sentences of text for the person who reads it
EXIT_GRACE_S = 60.0              # how long a worker that was asked to close may take to exit before it is killed
REPLY_TIMEOUT_S = 3000.0         # the supervisor's own deadline is the limit that matters; this ends a worker that stopped answering
MIN_LOAD_S = 1e-6                # a load the clock cannot resolve is reported as this, not as 0
PREFIX = "port| "
COUNTS = collections.Counter()   # the worker's tally of ttnn calls since the last request began (see CountingModule)
OPS_KEY = "ops"                  # the worker's reply field that carries the tally
DEVICE_OPS_FAILURE = "device-ops"          # the `op` of a failure of the device-op gate (orchard/port_repair.py reads it from the failure line)
MAX_OP_NAMES = 60                # names kept per request's tally
SCAN_NODES = 100_000             # objects visited by the weight scan
SCAN_DEPTH = 16                  # levels of containers and attributes the weight scan goes down
SCAN_TOP = 8                     # host tensors listed in the failure message, largest first
SCAN_DEVICE_NAMES = 20_000       # device tensor names kept to pair with a host tensor
PATH_KEY_MAX = 80                # characters of one dict key or attribute name in a path
HOST_MESSAGE_MAX = 1500          # characters of the load-gate failure message
HOST_SHARE_MAX = 0.5             # more than this share of the weight elements on the host fails the run
CONFIG_MAX = 1 << 20             # the snapshot's config.json is a few kilobytes
SOURCE_MAX = 1 << 21             # a port source file is read to this size for the static count
MARK = "PORT_CHECK"              # the start of the line the parent prints last; a port that prints it fails the run
FORGED = threading.Event()       # set when a line from the worker mentions MARK


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path) -> str:
    return sha_bytes(Path(path).read_bytes())


# The static count and the layer floor. This block is orchard/ttnn_calls.py between its `shared` markers, byte for byte
# (tests/test_ttnn_calls.py compares them), because this file runs from the stage directory and cannot import `orchard`.
# --- begin shared: ttnn_calls
import ast
import re

MATMUL_FAMILY = re.compile(r"matmul|linear|sdpa|scaled_dot_product|conv[123]d|conv_transpose|einsum")
# ttnn names that manage the device or build a shape. A call to one is not a computation on the device.
NOT_COMPUTE = frozenset({"open_device", "close_device", "open_mesh_device", "close_mesh_device", "set_fabric_config", "MeshShape",
                         "synchronize_device", "get_device_ids", "get_num_devices", "manage_device", "begin_trace_capture",
                         "end_trace_capture", "release_trace", "deallocate", "to_torch", "from_torch", "as_tensor", "to_device",
                         "from_device", "load_tensor", "dump_tensor", "to_dtype", "to_layout", "reshape", "typecast", "get_arch_name"})
LAYER_COUNT_KEYS = ("num_hidden_layers", "n_layer", "num_layers", "n_layers")
LAYER_COUNT_MAX = 10_000
_ZERO_NAMES = frozenset({"TYPE_CHECKING"})


def is_matmul(name: str) -> bool:
    """True when the dotted ttnn name (`ttnn.linear`, `ttnn.experimental.minimal_matmul`) is a matmul-family op."""
    tail = name.lower().rsplit(".", 1)[-1]
    if tail.startswith(("create_", "get_")) or "config" in tail or tail.endswith("attributes"):
        return False                     # a helper that builds a settings object or a name lookup does not run a projection
    return bool(MATMUL_FAMILY.search(name.lower()))


def is_compute(name: str) -> bool:
    """True when the call is an op that computes, and not device management, a shape or a copy between host and device."""
    return name.rsplit(".", 1)[-1] not in NOT_COMPUTE


def _chain(node):
    """The dotted name of a Name or Attribute chain (`a.b.c`), or None."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _constant_test(test):
    """True or False when an `if` test is a constant a plain reading can decide, else None."""
    if isinstance(test, ast.Constant):
        return bool(test.value)
    if isinstance(test, ast.Name) and test.id in _ZERO_NAMES:
        return False
    if isinstance(test, ast.Attribute) and test.attr in _ZERO_NAMES:
        return False
    return None


def static_calls(source: str) -> dict:
    """{"calls": {dotted name: count}, "error": None} for the reachable `ttnn.<op>(` calls of `source`; with a syntax error, {"calls": {}, "error": reason}.

    A name counts as ttnn when it is bound by `import ttnn`, `import ttnn as t`, `from ttnn import linear` or `x = ttnn.linear`, and the call is
    `ttnn.linear(...)`, `t.linear(...)`, `linear(...)` or `x(...)`. Calls that only manage the device (NOT_COMPUTE) are listed in `calls` too; use
    `is_compute` to leave them out."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
        return {"calls": {}, "error": f"{type(exc).__name__}: the file does not parse"}
    roots: dict[str, str] = {}          # local name -> the dotted ttnn name it stands for

    def bind(stmt):
        if isinstance(stmt, ast.Import):
            for a in stmt.names:
                if a.name == "ttnn" or a.name.startswith("ttnn."):
                    roots[a.asname or "ttnn"] = a.name if a.asname else "ttnn"
        elif isinstance(stmt, ast.ImportFrom) and stmt.level == 0 and (stmt.module == "ttnn" or (stmt.module or "").startswith("ttnn.")):
            for a in stmt.names:
                roots[a.asname or a.name] = f"{stmt.module}.{a.name}"
        elif isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            dotted = _chain(stmt.value)
            if dotted and dotted.split(".", 1)[0] in roots:
                head, _, rest = dotted.partition(".")
                roots[stmt.targets[0].id] = roots[head] + ("." + rest if rest else "")

    calls: dict[str, int] = {}

    def look(node):
        """The calls in an expression or in the parts of a statement that are not blocks of statements."""
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                dotted = _chain(sub.func)
                if dotted and dotted.split(".", 1)[0] in roots:
                    head, _, rest = dotted.partition(".")
                    name = roots[head] + ("." + rest if rest else "")
                    if name != "ttnn":
                        calls[name] = calls.get(name, 0) + 1

    def block(stmts):
        for stmt in stmts:
            bind(stmt)
            if isinstance(stmt, ast.If):
                look(stmt.test)
                decided = _constant_test(stmt.test)
                if decided is not False:
                    block(stmt.body)
                if decided is not True:
                    block(stmt.orelse)
            elif isinstance(stmt, ast.While) and _constant_test(stmt.test) is False:
                block(stmt.orelse)
            elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                for d in stmt.decorator_list:
                    look(d)
                if isinstance(stmt, ast.ClassDef):
                    for b in stmt.bases:
                        look(b)
                else:
                    for d in [*stmt.args.defaults, *[k for k in stmt.args.kw_defaults if k is not None]]:
                        look(d)
                block(stmt.body)
            elif hasattr(stmt, "body") or hasattr(stmt, "handlers"):          # for, while, with, try, match: the header, then each block
                for field, value in ast.iter_fields(stmt):
                    if isinstance(value, list) and value and isinstance(value[0], ast.stmt):
                        block(value)
                    elif field == "handlers":
                        for h in value:
                            block(h.body)
                    elif field == "cases":
                        for c in value:
                            block(c.body)
                    elif isinstance(value, ast.AST):
                        look(value)
                    elif isinstance(value, list):
                        for v in value:
                            if isinstance(v, ast.AST) and not isinstance(v, (ast.stmt, ast.excepthandler, ast.match_case)):
                                look(v)
                            elif isinstance(v, ast.withitem):
                                look(v)
            else:
                look(stmt)
            if isinstance(stmt, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
                return                       # what follows in this block cannot run

    # Function bodies run after the whole module has loaded, so an import or alias that comes later in the file still binds the names they use.
    # Bind every import and alias first (twice, so an alias of an alias settles), then count the calls in reading order.
    for _ in range(2):
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign)):
                bind(node)
    block(tree.body)
    return {"calls": dict(sorted(calls.items())), "error": None}


def layer_floor(config) -> tuple[int, int | None]:
    """(floor, layers): the number of matmul-family device calls one forward pass must make at least, and the layer count it came from (None
    when the config does not give one). Every decoder layer projects its input at least once, so a pass that touches the device for every layer
    makes at least one matmul or linear per layer. Without a layer count the floor is 1."""
    section = config.get("text_config") if isinstance(config, dict) and isinstance(config.get("text_config"), dict) else config
    if not isinstance(section, dict):
        return 1, None
    for key in LAYER_COUNT_KEYS:
        n = section.get(key)
        if type(n) is int and 0 < n <= LAYER_COUNT_MAX:
            return n, n
    kinds = section.get("layer_types")
    if isinstance(kinds, list) and 0 < len(kinds) <= LAYER_COUNT_MAX:
        return len(kinds), len(kinds)
    return 1, None
# --- end shared: ttnn_calls


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
    return scrub(text)[:REPLY_CHARS]


def scrub(text) -> str:
    """`text` with URL user information and token-shaped words removed, whole. A caller that cuts a text short scrubs first: a cut can leave the
    start of a credential without the `@` or the length that the patterns look for."""
    return _TOKEN.sub("[token removed]", _USERINFO.sub(r"\1", str(text)))


def port_frame(exc) -> str:
    """`[port_model.py:LINE: source line] ` for the innermost traceback frame that lies in the port directory, else "". Run 20 of the tiny Qwen4
    rehearsal: the reply held the exception text alone, a C++ backtrace for a device error, so the coder could not tell which line of its port
    raised it and rewrote the whole file each round. Only a file under PORT_DIR is named, so the template's own frames add nothing. The source
    line is the coder's own text: scrubbed of credentials, then cut to 120 characters (PR review of run 20's lessons: the first version cut first and
    left the start of a URL password in the failure text)."""
    found = None
    root = os.path.realpath(PORT_DIR)
    tb = getattr(exc, "__traceback__", None)
    while tb is not None:
        path = os.path.realpath(tb.tb_frame.f_code.co_filename)
        if path.startswith(root + os.sep):
            found = (os.path.relpath(path, root), tb.tb_lineno, path)
        tb = tb.tb_next
    if found is None:
        return ""
    import linecache
    source = scrub(" ".join(linecache.getline(found[2], found[1]).split()))[:120]       # scrubbed before the cut, never after
    return f"[{found[0]}:{found[1]}" + (f": {source}" if source else "") + "] "


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


def layer_arrays(outputs, ids):
    """`outputs` of the port's `layer_outputs` as a dict of float32 numpy arrays named layer_000, layer_001, ..., or a ValueError that says what is wrong."""
    import numpy as np
    if not isinstance(outputs, (list, tuple)) or not 0 < len(outputs) <= MAX_LAYERS:
        raise ValueError(f"layer_outputs must return a list of 1 to {MAX_LAYERS} arrays, got {type(outputs).__name__}"
                         + (f" of {len(outputs)}" if isinstance(outputs, (list, tuple)) else ""))
    arrays, total = {}, 0
    for k, x in enumerate(outputs):
        if hasattr(x, "detach"):                       # a torch tensor
            x = x.detach().float().cpu().numpy()
        elif not isinstance(x, np.ndarray):
            raise ValueError(f"layer {k} is a {type(x).__name__}; expected a torch tensor or a numpy array")
        if x.dtype.kind not in "fiub":
            raise ValueError(f"layer {k} has dtype {x.dtype}, which is not numeric")
        total += int(x.size)
        if total > MAX_NUMEL:
            raise ValueError(f"the layers hold more than {MAX_NUMEL} elements in all")
        arrays[f"layer_{k:03d}"] = np.asarray(x, dtype=np.float32)
    arrays["prompt_ids"] = np.asarray(ids, dtype=np.int64)
    return arrays


def save_layers(mod, model, ids) -> dict:
    """The worker's reply to the `layers` request: write evidence/layer-port.npz from the port's optional hook (a temporary file, then a rename)."""
    hook = getattr(mod, "layer_outputs", None)
    if hook is None:
        return {"ok": True, "layers": 0}
    import numpy as np
    arrays = layer_arrays(hook(model, ids), ids)
    LAYER_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = LAYER_FILE.with_name(LAYER_FILE.name + ".tmp")
    if os.path.lexists(tmp):
        os.unlink(tmp)
    # O_EXCL and O_NOFOLLOW: a link the port left at the temporary name is not followed (the check removed any earlier one before the worker started).
    with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644), "wb") as f:  # hardening: ok descriptor from os.open with O_EXCL|O_NOFOLLOW
        np.savez(f, **arrays)
    os.replace(tmp, LAYER_FILE)
    return {"ok": True, "layers": len(arrays) - 1}


# ---- the device-op count: a proxy for `ttnn` in the worker, and the scan of what `load` returned -------------------------------------------

class Counted:
    """A ttnn op that adds one to COUNTS[name] each time it is called, then calls the real one."""

    def __init__(self, fn, name):
        self.__dict__.update(_fn=fn, _name=name)

    def __call__(self, *args, **kwargs):
        COUNTS[self._name] += 1
        return self._fn(*args, **kwargs)

    def __getattr__(self, attr):                 # `ttnn.matmul.golden_function` and the like
        if attr in ("_fn", "_name"):
            raise AttributeError(attr)
        return getattr(self._fn, attr)


class CountingModule(types.ModuleType):
    """The real `ttnn` (or one of its submodules), with each op reached through it counted. Classes, constants and non-callables come back
    unchanged, so `isinstance(x, ttnn.Tensor)`, `ttnn.bfloat16` and `ttnn.TILE_LAYOUT` behave as they do on the real module."""

    def __init__(self, real, name):
        super().__init__(name)
        self.__dict__.update(_real=real, _cname=name)
        if hasattr(real, "__path__"):
            self.__path__ = real.__path__

    def __getattr__(self, attr):
        if attr in ("_real", "_cname"):
            raise AttributeError(attr)
        value = getattr(self._real, attr)
        full = f"{self._cname}.{attr}"
        if isinstance(value, types.ModuleType):
            return CountingModule(value, full)
        if inspect.isclass(value) or not callable(value):
            return value
        return Counted(value, full)

    def __dir__(self):
        return dir(self._real)


def take_counts() -> dict:
    """The tally since the last call, as {name: count} for the MAX_OP_NAMES most called names, and the counter cleared."""
    out = dict(COUNTS.most_common(MAX_OP_NAMES))
    COUNTS.clear()
    return out


def weight_kind(x):
    """("host" or "device", element count, shape, kind) for a tensor-like object, else None. `kind` is "torch", "numpy" or "ttnn". Torch and
    numpy tensors are host. A ttnn tensor is device unless it reports a storage type that is not on the device. Read from the type's module and
    attributes only: no import of torch or numpy."""
    t = type(x)
    root = (t.__module__ or "").split(".")[0]
    try:
        if root == "torch" and hasattr(x, "numel") and hasattr(x, "dtype"):
            return "host", int(x.numel()), shape_of(x), "torch"
        if root == "numpy" and t.__name__ == "ndarray":
            return "host", int(x.size), shape_of(x), "numpy"
        if root == "ttnn" and hasattr(x, "shape") and not inspect.isclass(x) and not callable(x):
            numel = 1
            for d in tuple(x.shape):
                numel *= int(d)
            where = "device"
            if hasattr(x, "storage_type"):
                where = "device" if "DEVICE" in str(x.storage_type()).upper() else "host"
            return where, numel, shape_of(x), "ttnn"
    except Exception:
        return None
    return None


def _printable(text, limit: int) -> str:
    """`text` as one printable line cut to `limit` characters, for a message built from the worker's reply."""
    text = text if isinstance(text, str) else "?"
    text = "".join(c if c.isprintable() else "?" for c in text)
    return text if len(text) <= limit else text[:limit - 3] + "..."


def shape_of(x) -> tuple:
    """The shape as at most 8 whole numbers, or () when the object does not give one."""
    try:
        return tuple(int(d) for d in tuple(x.shape))[:8]
    except Exception:
        return ()


def step_name(key) -> str:
    """One dict key as it appears in a path: a short string or number in brackets. Any other key type is named by its type, so no user code
    (a `__repr__`) runs while the walk builds a path."""
    if isinstance(key, str):
        key = "".join(c if c.isprintable() else "?" for c in key[:PATH_KEY_MAX]) + ("..." if len(key) > PATH_KEY_MAX else "")
        return f"[{key!r}]"
    if type(key) in (int, bool):
        return f"[{int(key)}]"
    return f"[<{type(key).__name__}>]"


def leaf_name(path: str) -> str:
    """The last step of a path, the name a host tensor and its device twin share (`'up_proj.weight'` or `.up_proj`)."""
    for mark in ("['", "."):
        if mark in path:
            tail = path.rsplit(mark, 1)[1]
            return tail.rstrip("']")
    return path


def scan_weights(*roots, labels=None) -> dict:
    """Walk what the port holds (dicts, lists, tuples, sets and object attributes, no modules, classes or functions) and total the weight
    elements by where they live. Each object is visited once (a visited-id set, so a cycle ends), at most SCAN_NODES objects and SCAN_DEPTH
    levels deep; `truncated` says a cap stopped the walk. Properties are never called. Besides the totals it keeps the SCAN_TOP largest host
    tensors with their paths from the roots (`labels` names the roots, default root0, root1; names and shapes only, never values) and, for
    each, the device tensor that has the same leaf name, if any."""
    labels = list(labels or []) + [f"root{i}" for i in range(len(roots))][len(labels or []):]
    seen, host, dev, n_host, n_dev, visited, cut = set(), 0, 0, 0, 0, 0, False
    stack = [(r, labels[i], 0) for i, r in reversed(list(enumerate(roots)))]
    top, tick, device_by_leaf = [], 0, {}              # top: a min-heap of (elements, -arrival, entry), at most SCAN_TOP long
    while stack and visited < SCAN_NODES:
        x, path, depth = stack.pop()
        if id(x) in seen or x is None or isinstance(x, (str, bytes, int, float, bool, types.ModuleType, types.FunctionType, type)):
            continue
        seen.add(id(x))
        visited += 1
        kind = weight_kind(x)
        if kind is not None:
            where, numel, shape, name = kind
            if where == "host":
                host, n_host, tick = host + numel, n_host + 1, tick + 1
                entry = {"path": path, "shape": list(shape), "elements": numel, "kind": name if name != "ttnn" else "ttnn-host"}
                if len(top) < SCAN_TOP:
                    heapq.heappush(top, (numel, -tick, id(entry), entry))
                elif (numel, -tick) > top[0][:2]:
                    heapq.heapreplace(top, (numel, -tick, id(entry), entry))
            else:
                dev, n_dev = dev + numel, n_dev + 1
                if len(device_by_leaf) < SCAN_DEVICE_NAMES:
                    device_by_leaf.setdefault(leaf_name(path), {"path": path, "shape": list(shape)})
            continue
        if depth >= SCAN_DEPTH:
            cut = cut or bool(children(x))             # a deeper level exists: the walk stops here and says so
            continue
        for step, child in reversed(children(x)):
            stack.append((child, path + step, depth + 1))
    rows = [e for _n, _t, _i, e in sorted(top, key=lambda t: (-t[0], -t[1]))]
    for e in rows:
        e["device"] = device_by_leaf.get(leaf_name(e["path"]))
    return {"host_elements": host, "device_elements": dev, "host_tensors": n_host, "device_tensors": n_dev, "visited": visited,
            "truncated": bool(stack) or cut, "top_host": rows}


def children(x) -> list:
    """(path step, child) pairs of a container or object, without calling any property or user code. Capped at SCAN_NODES entries."""
    try:
        if isinstance(x, dict):
            return [(step_name(k), v) for k, v in itertools.islice(x.items(), SCAN_NODES)]
        if isinstance(x, (list, tuple)):
            return [(f"[{i}]", v) for i, v in enumerate(itertools.islice(x, SCAN_NODES))]
        if isinstance(x, (set, frozenset)):
            return [("{...}", v) for v in itertools.islice(x, SCAN_NODES)]
        return [("." + str(k)[:PATH_KEY_MAX], v) for k, v in itertools.islice(vars(x).items(), SCAN_NODES)]
    except TypeError:
        return []


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
                sys.modules["ttnn"] = CountingModule(ttnn, "ttnn")       # the port imports this one: every op it calls is counted
                sys.path.insert(0, str(PORT_DIR))
                mod = importlib.import_module(PORT_MODULE)
                reply({"ok": True})
            elif op == "load":           # timed: only the port's own load
                COUNTS.clear()
                model = mod.load(req["model_dir"], device)
                reply({"ok": True, OPS_KEY: take_counts()})
            elif op == "scan":           # untimed, right after `load`: where the port's weights live
                reply({"ok": True, "weights": scan_weights(model, vars(mod), labels=["model", "port_module"])})
            elif op == "prefill":
                COUNTS.clear()
                reply({"ok": True, "id": argmax(mod.prefill(model, [int(i) for i in req["ids"]])), OPS_KEY: take_counts()})
            elif op == "decode":
                COUNTS.clear()
                reply({"ok": True, "id": argmax(mod.decode_step(model, int(req["token"]), int(req["position"]))), OPS_KEY: take_counts()})
            elif op == "layers":         # optional, untimed, after the comparison: per-layer hidden states for `layer-diff`
                reply(save_layers(mod, model, [int(i) for i in req["ids"]]))
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
            reply({"ok": False, "error": clip(f"{port_frame(exc)}{type(exc).__name__}: {exc}")})
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
        self.last = answer
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


class DeviceOps:
    """The parent's tally of the ttnn calls the worker reported, and the checks on it (issue 541). Each check raises PortFailed with a message
    that says what to do. The worker's numbers are taken as given: the parent cannot see inside the worker, so a port that falsifies them is
    not caught here, which the module's docstring states."""

    def __init__(self, floor: int, layers):
        self.floor, self.layers = floor, layers
        self.prefill_matmul = None
        self.decode_matmul_max = 0
        self.decode_trace_replays = 0
        self.total = collections.Counter()
        self.weights = None
        self.static_matmul = None

    @staticmethod
    def _clean(ops) -> dict:
        """The worker's tally with each entry type-checked: a name of at most 120 characters and a count from 0 to 10**9."""
        if not isinstance(ops, dict):
            return {}
        return {k: v for k, v in list(ops.items())[:MAX_OP_NAMES] if isinstance(k, str) and len(k) <= 120 and type(v) is int and 0 <= v <= 10 ** 9}

    @staticmethod
    def matmuls(ops: dict) -> int:
        return sum(v for k, v in ops.items() if is_matmul(k))

    @staticmethod
    def replays(ops: dict) -> int:
        return sum(v for k, v in ops.items() if k.rsplit(".", 1)[-1] == "execute_trace")

    def seen(self, ops: dict) -> str:
        top = sorted(ops.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
        return ", ".join(f"{k} x{v}" for k, v in top) if top else "none"

    def what_to_do(self) -> str:
        return ("Move the computation onto the device: in load(), upload each weight with ttnn.from_torch(w, dtype=ttnn.bfloat16, "
                "layout=ttnn.TILE_LAYOUT, device=mesh_device) (or ttnn.as_tensor) and keep the ttnn tensor; in prefill and decode_step, run each "
                "projection with ttnn.linear or ttnn.matmul on those tensors. stages/2/cheat_sheet.md lists the call forms. A host torch port with "
                "`import ttnn` computes on the CPU, so the run does not show the chip works, and it does not count. An import, a comment, a "
                "string and code that is not reached make no calls.")

    def check_static(self) -> None:
        """Before the device is touched: some file under the port directory must have a reachable matmul-family ttnn call."""
        names, errors = collections.Counter(), 0
        for f in sorted(PORT_DIR.rglob("*.py")):
            if not f.is_file() or f.is_symlink():
                continue
            with open(f, "rb") as fh:             # hardening: ok regular file under the port directory (links are refused first), bounded read
                text = fh.read(SOURCE_MAX + 1).decode("utf-8", "replace")
            found = static_calls(text)
            errors += found["error"] is not None
            names.update(found["calls"])
        self.static_matmul = sum(v for k, v in names.items() if is_matmul(k) and is_compute(k))
        if self.static_matmul or errors:          # a file that does not parse is reported by the worker's import, which names the line
            return
        listed = ", ".join(f"{k} x{v}" for k, v in sorted(names.items())[:8]) or "none"
        raise PortFailed(f"{DEVICE_OPS_FAILURE}: no source file under stages/2/port/ makes a reachable ttnn matmul-family call (ttnn.matmul, "
                         f"ttnn.linear and the like). ttnn calls found: {listed}. " + self.what_to_do(), op=DEVICE_OPS_FAILURE)

    def check_weights(self, scan) -> None:
        """After load: the weights the port holds must be mostly ttnn tensors. A scan that finds no tensor at all is recorded as such and the
        matmul counts decide."""
        if not isinstance(scan, dict):
            return
        host, dev = scan.get("host_elements"), scan.get("device_elements")
        if type(host) is not int or type(dev) is not int or host < 0 or dev < 0:
            return
        self.weights = {k: scan.get(k) for k in ("host_elements", "device_elements", "host_tensors", "device_tensors", "visited", "truncated")}
        if host + dev and host / (host + dev) > HOST_SHARE_MAX:
            raise PortFailed(f"{DEVICE_OPS_FAILURE}: after load() the port holds {host} weight elements as host torch or numpy tensors and "
                             f"{dev} as ttnn tensors; at most {int(HOST_SHARE_MAX * 100)}% may stay on the host. " + self.host_listing(scan)
                             + self.what_to_do(), op="load")

    @staticmethod
    def host_listing(scan: dict) -> str:
        """The largest host-held tensors with their paths (`model` is what load() returned), the count of the rest, and the one thing to do.
        Every field of the worker's list is type-checked here; a bad row is skipped. The text is cut to HOST_MESSAGE_MAX characters, the
        closing instruction kept whole."""
        rows = scan.get("top_host")
        parts = []
        for r in (rows if isinstance(rows, list) else [])[:SCAN_TOP]:
            if not isinstance(r, dict) or type(r.get("elements")) is not int or not isinstance(r.get("path"), str):
                continue
            shape = r.get("shape")
            shape = "x".join(str(d) for d in shape[:8] if type(d) is int) if isinstance(shape, list) else "?"
            twin = r.get("device")
            twin = (f"; the device copy is {_printable(twin.get('path'), 90)}" if isinstance(twin, dict) and isinstance(twin.get("path"), str) else "")
            parts.append(f"{_printable(r['path'], 110)} shape {shape} = {r['elements']} elements, {_printable(r.get('kind'), 10)}{twin}")
        if not parts:
            return ""
        total = scan.get("host_tensors")
        rest = total - len(parts) if type(total) is int and total > len(parts) else 0
        head = "Largest host copies, from the value load() returned (named `model` here): " + " | ".join(parts)
        head += f" | and {rest} more host tensors" if rest else ""
        head += ". The walk was cut short (walk truncated), so more may exist." if scan.get("truncated") is True else "."
        tail = (" Delete or do not store the host copy: after ttnn.from_torch(w, ...) keep only the ttnn tensor; see stages/2/decoder_idioms.md. ")
        return head[:HOST_MESSAGE_MAX - len(tail)] + tail

    def add_load(self, ops) -> None:
        self.total.update(self._clean(ops))

    def add_prefill(self, ops) -> None:
        ops = self._clean(ops)
        self.total.update(ops)
        self.prefill_matmul = self.matmuls(ops)
        self.prefill_seen = self.seen(ops)

    def check_prefill(self) -> None:
        if self.prefill_matmul < self.floor:
            of = f"the {self.layers} layers of the config" if self.layers else "the model"
            raise PortFailed(f"{DEVICE_OPS_FAILURE}: the first prefill made {self.prefill_matmul} matmul-family ttnn calls; a forward pass over {of} "
                             f"makes at least {self.floor} (one ttnn.matmul or ttnn.linear per layer). ttnn calls seen in that prefill: "
                             f"{self.prefill_seen}. " + self.what_to_do(), op="prefill")

    def add_decode(self, ops) -> None:
        ops = self._clean(ops)
        self.total.update(ops)
        self.decode_matmul_max = max(self.decode_matmul_max, self.matmuls(ops))
        self.decode_trace_replays += self.replays(ops)

    def check_decode(self) -> None:
        """Some decode step must make the floor's worth of matmul calls. A step that replays a captured trace makes none itself: the captured
        pass (counted wherever it ran) must have made the floor's worth, which the total over the whole run shows."""
        if self.decode_matmul_max >= self.floor:
            return
        if self.decode_trace_replays and self.matmuls(self.total) >= 2 * self.floor:
            return
        raise PortFailed(f"{DEVICE_OPS_FAILURE}: no decode step made as many as {self.floor} matmul-family ttnn calls (the most in one step was "
                         f"{self.decode_matmul_max}; trace replays: {self.decode_trace_replays}). Decoding a token runs every layer, so a step makes "
                         f"at least one ttnn.matmul or ttnn.linear per layer. " + self.what_to_do(), op="decode")

    def record(self) -> dict:
        top = sorted(self.total.items(), key=lambda kv: (-kv[1], kv[0]))[:12]
        return {"floor": self.floor, "layers": self.layers, "prefill_matmul_calls": self.prefill_matmul, "decode_matmul_calls_max": self.decode_matmul_max,
                "decode_trace_replays": self.decode_trace_replays, "matmul_calls_total": self.matmuls(self.total),
                "static_matmul_calls": self.static_matmul, "weights": self.weights, "top_calls": dict(top)}


def read_config(model_dir: str) -> dict:
    """The snapshot's config.json as a dict, or {} when it cannot be read (the floor is then 1)."""
    try:
        with open(Path(model_dir) / "config.json", "rb") as fh:      # hardening: ok the supervisor's own snapshot path; a bounded read in a try
            data = json.loads(fh.read(CONFIG_MAX + 1)[:CONFIG_MAX])
    except (OSError, ValueError, RecursionError):
        return {}
    return data if isinstance(data, dict) else {}


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
    for stale in (RECORD, LAYER_FILE, LAYER_FILE.with_name(LAYER_FILE.name + ".tmp")):
        if stale.exists() or stale.is_symlink():
            stale.unlink()

    floor, layers = layer_floor(read_config(model_dir))
    device_ops = DeviceOps(floor, layers)
    device_ops.check_static()          # the early reject: no reachable ttnn matmul-family call anywhere in the port, so nothing to run on the device

    w = Worker()
    try:
        w.ask(op="init", n_chips=len(chips))
        t0 = time.monotonic()
        w.ask(op="load", model_dir=model_dir)
        device_ops.add_load(w.last.get(OPS_KEY))       # a warm-up pass that captures the decode trace runs here
        load_s = max(time.monotonic() - t0, MIN_LOAD_S)     # full precision; a rounded value could be 0.0, which the gate refuses
        print(f"loaded in {load_s}s", flush=True)
        w.ask(op="scan")                 # untimed: where the weights the port holds live
        device_ops.check_weights(w.last.get("weights"))
        # Teacher forced: at position i the port's argmax is taken, then the reference id i is fed in, never the port's own.
        chip = [w.ask(op="prefill", ids=prompt)]
        device_ops.add_prefill(w.last.get(OPS_KEY))
        device_ops.check_prefill()       # the first forward pass is the cheapest place to learn it never touched the device
        for i in range(1, n):
            chip.append(w.ask(op="decode", token=ref[i - 1], position=len(prompt) + i - 1))
            device_ops.add_decode(w.last.get(OPS_KEY))
        device_ops.check_decode()
        # Free run: a fresh prefill, then the port's own greedy ids.
        free = [w.ask(op="prefill", ids=prompt)]
        for k in range(1, FREE_RUN_TOKENS):
            free.append(w.ask(op="decode", token=free[-1], position=len(prompt) + k - 1))
        agreement = sum(1 for a, b in zip(ref, chip) if a == b) / n
        layers_note, n_layers = None, 0
        if agreement < LAYER_DIAG_BELOW:          # advice for the coder, after everything the record needs; nothing in it can change the record
            try:
                w.ask(op="layers", ids=prompt)
                n_layers = int(w.last.get("layers") or 0)
                layers_note = ("layer outputs: none (port_model defines no layer_outputs(model, ids))" if not n_layers
                               else f"layer outputs: {n_layers} layers saved")
            except PortFailed as exc:
                if "did not answer" in str(exc):      # the worker is gone or stuck: the same failure as for any other request
                    raise
                layers_note = f"layer outputs not written: {str(exc)[:600]}"
        w.ask(op="close")
    finally:
        w.stop()
    refuse_links()                  # again: a link made while the port ran
    if FORGED.is_set():
        raise PortFailed(f"the port printed a line that mentions {MARK}: only this file prints that line, so the run is refused")

    if layers_note is not None:
        print(layers_note, flush=True)
        if LAYER_FILE.is_file():
            print(LAYER_MARK, sha_file(LAYER_FILE), n_layers, flush=True)      # the worker is stopped: the file is final
    record = {"schema": SCHEMA,
              "reference": {"path": REFERENCE, "sha256": sha_file(RUN_DIR / REFERENCE)},
              "prompt": {"path": PROMPT, "sha256": sha_file(RUN_DIR / PROMPT)},
              "reference_ids": ref, "chip_ids": chip,
              "top1_agreement": agreement,
              "chips": chips, "load_s": load_s, "free_run_ids": free, "free_run_text": free_run_text(model_dir, free),
              "device_ops": device_ops.record(),
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
