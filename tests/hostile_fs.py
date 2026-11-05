# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Hostile directory structures, for every reader and writer that works on a tree something else controls.

tests/hostile_inputs.py holds the corpus of hostile CONTENT and hostile LEAF files (a FIFO, a link, a 50 MB string) and the registry of
readers that take one file. This module holds what that corpus cannot say: the shape of the PATH. Reviews of this repository kept finding
the same five mistakes in code that already used `O_NOFOLLOW`:

  1. `O_NOFOLLOW` protects only the last component, so a link swapped in higher up is followed.
  2. The root is pinned by descriptor but a later listing, read or write goes by pathname, so a root renamed and replaced by a link
     after the pin redirects it.
  3. A FIFO named like the expected file reports size 0, passes a size check and blocks the read.
  4. A lock or temp file opened with mode `w` by path follows a link and truncates its target before the lock is taken.
  5. The opposite error: a rule that refuses or mis-measures a legitimate link, such as a Hugging Face cache blob (a link from
     `snapshots/<rev>/<file>` into `blobs/`).

The cases below build each shape under a temporary directory. `PATH_READERS` registers the readers and writers that must survive them
(`family` says which cases apply and what must happen). `run_reader` runs one reader against all its cases in a child process with an
address-space limit and a per-case alarm, and returns a verdict per case. tests/test_hostile_fs_readers.py turns the verdicts into
tests; a reader that fails today is listed in `KNOWN_DEFECTS` with the issue that tracks it and is marked xfail(strict=True).

A new reader of a tree that something else controls adds one `PathReader` line and passes, or says why it cannot.

Verdicts (what the reader did):
    original   it returned or wrote what the plain layout holds
    refused    it raised a safeio refusal or the reader's own named error
    leaked     it returned content from outside the root, or changed a file outside the root
    hang       it did not finish inside the alarm
    crash      it raised something else (a bare OSError that is not a refusal counts as refused; see `REFUSAL_NAMES`)
"""
from __future__ import annotations

import json
import os
import signal
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parent.parent
TESTS = Path(__file__).resolve().parent
ALARM_S = 4
ADDRESS_SPACE = 2 << 30
OK_BYTES = b'{"ok": 1}'
DECOY_BYTES = b'{"ok": 0, "decoy": true}'
VICTIM_BYTES = b"precious\n"
HUGE_SPARSE = 1 << 30                       # a sparse file of 1 GiB: it costs no disk and is far over every cap used here

# Class names of the repository's own refusals. A reader's own named error counts as a refusal; a bare KeyError or TypeError does not.
REFUSAL_NAMES = frozenset({
    "SafeReadError", "ReadFailed", "PathRefused", "NotRegular", "TooLarge", "ChangedWhileReading", "NotText", "NotJson", "WrongType",
    "BadValue", "BadHeader", "IncompleteScan", "ResidencyError", "LedgerLocked", "LedgerCorrupt",
})


# ------------------------------------------------------------------------------------------------------------------- layout

def layout(base: Path) -> dict:
    """The plain layout under `base`: `root/` holds `sub/data.json` (OK_BYTES) and a small tree; `outside/` holds a decoy of the same name
    and `victim.txt`. Returns the paths. Cases start from this and change one thing."""
    root, outside = base / "root", base / "outside"
    (root / "sub").mkdir(parents=True)
    (outside / "sub").mkdir(parents=True)
    (root / "sub" / "data.json").write_bytes(OK_BYTES)
    (root / "sub" / "other.txt").write_bytes(b"x")
    (outside / "sub" / "data.json").write_bytes(DECOY_BYTES)
    (outside / "sub" / "DECOY_MARK").write_bytes(b"x")
    (outside / "victim.txt").write_bytes(VICTIM_BYTES)
    return {"base": base, "root": root, "outside": outside, "rel": "sub/data.json", "parts": ("sub", "data.json"),
            "file": root / "sub" / "data.json", "lock": root / "ledger.jsonl.lock", "victim": outside / "victim.txt",
            "between": lambda: None}


def _fifo_named(env):
    env["file"].unlink()
    os.mkfifo(env["file"])


def _leaf_link(env):
    env["file"].unlink()
    env["file"].symlink_to(env["outside"] / "sub" / "data.json")


def _ancestor_link(env):
    (env["root"] / "sub" / "data.json").unlink()
    (env["root"] / "sub" / "other.txt").unlink()
    (env["root"] / "sub").rmdir()
    (env["root"] / "sub").symlink_to(env["outside"] / "sub")


def _root_is_link(env):
    link = env["base"] / "rootlink"
    link.symlink_to(env["root"])
    env["root"] = link                                  # the reader is handed the link as its root


def _swap_after_pin(env):
    root, outside = env["root"], env["outside"]

    def swap():
        os.rename(root, str(root) + ".orig")            # the pinned directory moves away
        os.symlink(outside, root)                       # and a link to the decoy tree takes its name
    env["between"] = swap


def _dir_for_file(env):
    env["file"].unlink()
    env["file"].mkdir()
    (env["file"] / "inner.txt").write_bytes(b"x")


def _sparse_huge(env):
    env["file"].unlink()
    with open(env["file"], "wb") as fh:
        fh.truncate(HUGE_SPARSE)


def _lock_link(env):
    env["lock"].symlink_to(env["victim"])


def _deep_tree(env):
    cur = env["root"] / "deep"
    cur.mkdir()
    for _ in range(300):
        cur = cur / "d"
        cur.mkdir()


def _long_names(env):
    d = env["root"] / "wide"
    d.mkdir()
    for i in range(300):
        (d / (f"{i:04d}" + "n" * 200)).write_bytes(b"")


def _tree_with_links(env):
    (env["root"] / "sub" / "to_outside").symlink_to(env["outside"] / "sub")      # a link to a directory inside the tree
    (env["root"] / "sub" / "to_file").symlink_to(env["outside"] / "victim.txt")


def _tiny_safetensors(path: Path) -> None:
    header = json.dumps({"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 8)


def _hf_inside(env):
    """A Hugging Face cache snapshot: `snapshot/model.safetensors` is a link into `blobs/` inside the same cache root."""
    cache = env["root"] / "models--x"
    (cache / "blobs").mkdir(parents=True)
    (cache / "snapshot").mkdir()
    _tiny_safetensors(cache / "blobs" / "abc123")
    (cache / "snapshot" / "model.safetensors").symlink_to("../blobs/abc123")
    env["snapshot"] = cache / "snapshot"


def _hf_outside(env):
    """The same snapshot, but the link points outside the cache root."""
    cache = env["root"] / "models--x"
    (cache / "snapshot").mkdir(parents=True)
    _tiny_safetensors(env["outside"] / "elsewhere.safetensors")
    (cache / "snapshot" / "model.safetensors").symlink_to(env["outside"] / "elsewhere.safetensors")
    env["snapshot"] = cache / "snapshot"


def _sidecar_link(env):
    """A weights-looking file in the model directory that is a link to a weights file outside the model directory."""
    (env["root"] / "extra.gguf").symlink_to(env["outside"] / "victim.txt")


@dataclass(frozen=True)
class Case:
    name: str
    build: Callable[[dict], None]
    why: str                                   # the review finding or the rule this case stands for


CASES: dict[str, Case] = {c.name: c for c in [
    Case("plain", lambda env: None, "the control: nothing hostile, the reader must do its job"),
    Case("leaf_link", _leaf_link, "the last component is a link to a file outside the root"),
    Case("ancestor_link", _ancestor_link, "a directory on the way is a link to a tree outside the root (O_NOFOLLOW on the last part alone misses it)"),
    Case("root_is_link", _root_is_link, "the root path handed in is itself a link"),
    Case("swap_after_pin", _swap_after_pin, "the root is renamed away and a link to a decoy tree takes its name after the reader pinned it"),
    Case("fifo_named_file", _fifo_named, "a FIFO with the file's name: size 0 passes a size check and a read blocks"),
    Case("dir_for_file", _dir_for_file, "a directory where a file is expected"),
    Case("sparse_huge", _sparse_huge, "a 1 GiB sparse file: far over any cap, almost free to create"),
    Case("lock_link", _lock_link, "the lock file is a link to a victim file: opening it with mode w truncates the victim"),
    Case("deep_tree", _deep_tree, "300 nested directories"),
    Case("long_names", _long_names, "300 entries with 200-character names"),
    Case("tree_with_links", _tree_with_links, "links to a directory and a file inside the tree: reported, never entered"),
    Case("sidecar_link", _sidecar_link, "a weights-looking file (.gguf) in the model directory is a link to a file outside it: its size must not be listed (issue 464)"),
    Case("hf_blob_inside", _hf_inside, "the Hugging Face cache layout: a snapshot link into blobs/ inside the cache root MUST be accepted"),
    Case("hf_blob_outside", _hf_outside, "a snapshot link that leaves the cache root MUST be refused where the reader claims containment"),
]}


# ------------------------------------------------------------------------------------------------------------------- readers

@dataclass(frozen=True)
class PathReader:
    """One reader or writer of a tree something else controls.

    `call(env)` does the work and returns a short string describing what it saw (`"original"` or `"decoy"` for a read of data.json, a
    sorted list of names for a scan, `"written"` for a write) or raises. `family` picks the cases that apply and the verdict each must
    get (`EXPECT`). `claims_pin` says the reader promises that a root swapped after the pin is not followed."""
    name: str
    family: str
    call: Callable[[dict], str]
    claims_pin: bool = True


def _label(data: bytes) -> str:
    if data == OK_BYTES:
        return "original"
    if data == DECOY_BYTES:
        return "decoy"
    return "other"


def _pinned(fn):
    def call(env):
        from orchard import safeio
        with safeio.pinned_root(env["root"]) as pin:
            env["between"]()
            return fn(pin, env)
    return call


def _r_read_bytes_anchor(pin, env):
    from orchard import safeio
    return _label(safeio.read_bytes(env["rel"], max_bytes=1 << 20, anchor=pin.fd))


def _r_read_beneath(pin, env):
    from orchard import safeio
    return _label(safeio.read_beneath(pin.fd, env["parts"], max_bytes=1 << 20))


def _r_read_json_pin(pin, env):
    return _json_label(pin.read_json(env["parts"], max_bytes=1 << 20))


def _json_label(value) -> str:
    if value == {"ok": 1}:
        return "original"
    if value == {"ok": 0, "decoy": True}:
        return "decoy"
    return "other"


def _r_sha256_anchor(pin, env):
    import hashlib
    from orchard import safeio
    got = safeio.sha256_file(env["rel"], max_bytes=1 << 20, anchor=pin.fd)
    return "original" if got == hashlib.sha256(OK_BYTES).hexdigest() else ("decoy" if got == hashlib.sha256(DECOY_BYTES).hexdigest() else "other")


def _r_list_beneath(pin, env):
    from orchard import safeio
    with safeio.list_beneath(pin.fd, ("sub",)) as it:
        names = sorted(e.name for e in it)
    return "decoy" if "DECOY_MARK" in names else "original"


def _r_scan_beneath(pin, env):
    from orchard import safeio
    seen: list[str] = []
    safeio.scan_beneath(pin, lambda entry: seen.append(str(getattr(entry, "name", entry))))
    return "decoy" if "DECOY_MARK" in seen else "original"


def _r_write_beneath(pin, env):
    from orchard import safeio
    safeio.write_beneath(pin.fd, ("sub", "written.txt"), b"w")
    return "written"


def _r_read_bytes_path(env):
    from orchard import safeio
    return _label(safeio.read_bytes(env["file"], max_bytes=1 << 20))


def _r_read_json_path(env):
    from orchard import safeio
    return _json_label(safeio.read_json(env["file"], max_bytes=1 << 20))


def _r_atomic_write_path(env):
    from orchard import safeio
    safeio.atomic_write(env["root"] / "sub" / "data.json", b"new")
    return "written"


def _r_ledger_lock(env):
    from orchard import ledger
    led = ledger.Ledger(env["root"] / "ledger.jsonl")
    led.close()
    return "original"


def _r_cache_lock(env):
    from orchard import cache_lock
    with cache_lock.entry_lock(env["root"], "ledger.jsonl", exclusive=True):
        pass
    return "original"


def _r_residency_headers(env):
    from orchard import residency
    got = residency.read_headers(env["snapshot"])
    return "original" if [t.name for t in got.tensors] == ["w"] else "other"


def _r_compare_sidecars(env):
    import importlib.util
    path = REPO / "orchard" / "skills" / "delta-triage-templates" / "delta_triage.py"
    spec = importlib.util.spec_from_file_location("delta_triage_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    empty = env["base"] / "empty_base"
    empty.mkdir()
    got = mod.compare_sidecars(env["root"], empty, set())
    listed = [e["file"] for e in got["sidecars"]] + [e["file"] for e in got["ignored_files"]]
    return "decoy" if "extra.gguf" in listed else "original"


PATH_READERS: list[PathReader] = [
    PathReader("safeio.read_bytes(anchor=pin)", "pinned_read", _pinned(_r_read_bytes_anchor)),
    PathReader("safeio.read_beneath", "pinned_read", _pinned(_r_read_beneath)),
    PathReader("Pin.read_json", "pinned_read", _pinned(_r_read_json_pin)),
    PathReader("safeio.sha256_file(anchor=pin)", "pinned_read", _pinned(_r_sha256_anchor)),
    PathReader("safeio.list_beneath", "pinned_list", _pinned(_r_list_beneath)),
    PathReader("safeio.scan_beneath", "scan", _pinned(_r_scan_beneath)),
    PathReader("safeio.write_beneath", "pinned_write", _pinned(_r_write_beneath)),
    PathReader("safeio.read_bytes(path)", "path_read", _r_read_bytes_path, claims_pin=False),
    PathReader("safeio.read_json(path)", "path_read", _r_read_json_path, claims_pin=False),
    PathReader("safeio.atomic_write(path)", "path_write", _r_atomic_write_path, claims_pin=False),
    PathReader("ledger.Ledger lock file", "lock_open", _r_ledger_lock, claims_pin=False),
    PathReader("cache_lock.entry_lock", "lock_open", _r_cache_lock, claims_pin=False),
    PathReader("residency.read_headers", "cache_headers", _r_residency_headers, claims_pin=False),
    PathReader("delta_triage.compare_sidecars", "sidecar_scan", _r_compare_sidecars, claims_pin=False),
]

# Deliberately unsafe readers: each makes one of the review mistakes, so the harness is seen to fail (a check that has never been seen
# to fail is a claim). tests/test_hostile_fs_readers.py asserts which cases each one is caught on.
def _c_naive_open(env):
    with open(env["file"], "rb") as fh:                      # mistake 3: follows a link, blocks on a FIFO, reads without a bound
        return _label(fh.read())


def _c_nofollow_last_only(env):
    fd = os.open(env["file"], os.O_RDONLY | os.O_NOFOLLOW)   # mistake 1: only the last component is protected
    try:
        return _label(os.read(fd, 1 << 20))
    finally:
        os.close(fd)


def _c_path_after_pin(env):
    from orchard import safeio
    with safeio.pinned_root(env["root"]):
        env["between"]()
        with open(os.path.join(env["root"], env["rel"]), "rb") as fh:       # mistake 2: pinned, then reads by pathname
            return _label(fh.read())


CONTROLS: list[PathReader] = [
    PathReader("control: builtin open", "path_read", _c_naive_open, claims_pin=False),
    PathReader("control: O_NOFOLLOW on the last part only", "pinned_read", _c_nofollow_last_only),
    PathReader("control: pin, then read by pathname", "pinned_read", _c_path_after_pin),
]

# family -> case -> the verdicts that are right. Anything else is a failure of the reader.
READ_BAD = {"leaked", "hang", "crash"}
EXPECT: dict[str, dict[str, set[str]]] = {
    "pinned_read": {"plain": {"original"}, "leaf_link": {"refused"}, "ancestor_link": {"refused"}, "root_is_link": {"refused"},
                    "swap_after_pin": {"original", "refused"}, "fifo_named_file": {"refused"}, "dir_for_file": {"refused"},
                    "sparse_huge": {"refused"}},
    "path_read": {"plain": {"original"}, "leaf_link": {"refused"}, "fifo_named_file": {"refused"}, "dir_for_file": {"refused"},
                  "sparse_huge": {"refused"}},
    "pinned_list": {"plain": {"original"}, "ancestor_link": {"refused"}, "root_is_link": {"refused"}, "swap_after_pin": {"original", "refused"}},
    "scan": {"plain": {"original"}, "tree_with_links": {"original"}, "swap_after_pin": {"original", "refused"}, "root_is_link": {"refused"},
             "deep_tree": {"original", "refused"}, "long_names": {"original", "refused"}},
    "pinned_write": {"plain": {"written"}, "ancestor_link": {"refused"}, "swap_after_pin": {"written", "refused"}},
    "path_write": {"plain": {"written"}, "leaf_link": {"written", "refused"}},
    "lock_open": {"plain": {"original"}, "lock_link": {"original", "refused"}},
    "cache_headers": {"hf_blob_inside": {"original"}, "hf_blob_outside": {"refused"}},
    "sidecar_scan": {"plain": {"original"}, "sidecar_link": {"original", "refused"}},
}


# ------------------------------------------------------------------------------------------------------------------- running

class _Alarm(Exception):
    pass


def _verdict(reader: PathReader, case_name: str, base: Path) -> dict:
    env = layout(base)
    env["case"] = case_name
    CASES[case_name].build(env)
    victim_before = env["victim"].read_bytes() if env["victim"].exists() else None
    outside_files_before = sorted(p.name for p in (env["outside"] / "sub").iterdir()) if (env["outside"] / "sub").is_dir() else []

    def on_alarm(signum, frame):
        raise _Alarm()
    signal.signal(signal.SIGALRM, on_alarm)
    signal.alarm(ALARM_S)
    got, detail = None, ""
    try:
        got = reader.call(env)
    except _Alarm:
        got, detail = "hang", f"no answer in {ALARM_S} s"
    except MemoryError:
        got, detail = "crash", "MemoryError (the address-space limit was reached)"
    except BaseException as exc:                                                       # the verdict names what happened
        name = type(exc).__name__
        got = "refused" if (name in REFUSAL_NAMES or (isinstance(exc, OSError) and name in ("OSError", "FileExistsError", "NotADirectoryError", "IsADirectoryError", "PermissionError", "FileNotFoundError"))) else "crash"
        detail = f"{name}: {str(exc)[:120]}"
    finally:
        signal.alarm(0)
    # Effects outside the root count as a leak whatever the reader returned.
    if env["victim"].exists() and env["victim"].read_bytes() != victim_before:
        got, detail = "leaked", "the file outside the root changed"
    elif (env["outside"] / "sub").is_dir() and sorted(p.name for p in (env["outside"] / "sub").iterdir()) != outside_files_before:
        got, detail = "leaked", "a file appeared outside the root"
    elif (env["outside"] / "sub" / "data.json").exists() and (env["outside"] / "sub" / "data.json").read_bytes() != DECOY_BYTES:
        got, detail = "leaked", "data.json outside the root changed"
    elif got == "decoy":
        got, detail = "leaked", "the reader returned content from outside the root"
    return {"case": case_name, "verdict": got, "detail": detail}


def child_main(reader_index: int, tmp: str, controls: bool = False) -> None:
    """Run reader number `reader_index` (of `CONTROLS` when `controls`) against every case of its family; print the verdicts as one JSON line."""
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (ADDRESS_SPACE, ADDRESS_SPACE))
    sys.path.insert(0, str(REPO))
    reader = (CONTROLS if controls else PATH_READERS)[reader_index]
    out = []
    for n, case_name in enumerate(EXPECT[reader.family]):
        base = Path(tmp) / f"{reader_index}-{n}"
        base.mkdir()
        out.append(_verdict(reader, case_name, base))
    print("VERDICTS " + json.dumps(out))


def run_reader(index: int, tmp: Path, *, timeout: float = 120, controls: bool = False) -> list[dict]:
    """Verdicts for `PATH_READERS[index]` (or `CONTROLS[index]`) against every case of its family, from a child process with an address-space limit."""
    code = f"import sys; sys.path.insert(0, {str(TESTS)!r}); import hostile_fs; hostile_fs.child_main({index}, {str(tmp)!r}, {controls!r})"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=timeout)
    for line in done.stdout.splitlines():
        if line.startswith("VERDICTS "):
            return json.loads(line[len("VERDICTS "):])
    return [{"case": "child", "verdict": "crash", "detail": (done.stderr or done.stdout)[-300:]}]


def assert_reader_safe(index: int, tmp: Path) -> None:
    """The helper a new reader's test calls: every case of its family must get one of the verdicts that are right, or this says which
    case, what the reader did, and why the case exists."""
    reader = PATH_READERS[index]
    bad = []
    for v in run_reader(index, tmp):
        want = EXPECT[reader.family].get(v["case"], set())
        if v["verdict"] not in want:
            bad.append(f"{v['case']}: got {v['verdict']} (want {sorted(want)}); {v['detail']}; case: {CASES[v['case']].why if v['case'] in CASES else '?'}")
    assert not bad, f"{reader.name} fails the hostile layouts:\n  " + "\n  ".join(bad)
