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
import stat
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
REVISION = "0123456789abcdef0123456789abcdef01234567"      # a snapshot directory is named by a commit hash
HUGE_SPARSE = 1 << 30                       # a sparse file of 1 GiB: it costs no disk and is far over every cap used here

# Class names of the repository's own refusals. A reader's own named error counts as a refusal; a bare KeyError or TypeError does not.
REFUSAL_NAMES = frozenset({
    "SafeReadError", "ReadFailed", "PathRefused", "NotRegular", "TooLarge", "ChangedWhileReading", "NotText", "NotJson", "WrongType",
    "BadValue", "BadHeader", "IncompleteScan", "ResidencyError", "LedgerLocked", "LedgerLockRefused", "LedgerCorrupt",
    "Unreadable",                                    # orchard/drop_check.py: a release file that cannot be read safely
})


# ------------------------------------------------------------------------------------------------------------------- layout

def _cache_lock_path(root: Path) -> Path:
    sys.path.insert(0, str(REPO))
    from orchard import cache_lock
    return cache_lock.lock_path(root, "ledger.jsonl")


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
            "file": root / "sub" / "data.json", "lock": root / "ledger.jsonl.lock", "cache_lock": _cache_lock_path(root), "victim": outside / "victim.txt",
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
    """Links at the exact lock names the readers open: the ledger's `<name>.lock` and cache_lock's `.<name>.lock` (taken from its own
    `lock_path`, so a change of naming there cannot leave this case opening a fresh file)."""
    env["lock"].symlink_to(env["victim"])
    env["cache_lock"].symlink_to(env["victim"])


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
    """A Hugging Face cache snapshot: `snapshots/<revision>/model.safetensors` is a link into `blobs/` inside the same cache root."""
    cache = env["root"] / "models--x"
    snap = cache / "snapshots" / REVISION
    (cache / "blobs").mkdir(parents=True)
    snap.mkdir(parents=True)
    _tiny_safetensors(cache / "blobs" / "abc123")
    (snap / "model.safetensors").symlink_to("../../blobs/abc123")           # the layout huggingface_hub writes
    env["snapshot"] = snap


def _hf_outside(env):
    """The same snapshot, but the link points outside the cache root."""
    snap = env["root"] / "models--x" / "snapshots" / REVISION
    snap.mkdir(parents=True)
    _tiny_safetensors(env["outside"] / "elsewhere.safetensors")
    (snap / "model.safetensors").symlink_to(env["outside"] / "elsewhere.safetensors")
    env["snapshot"] = snap


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


def classify_listing(names) -> str:
    """The verdict for a listing of `sub/`: `decoy` when it shows the outside tree, `original` when it shows the real `data.json`, and `empty`
    otherwise. An empty or wrong listing is never a pass: a reader that lists nothing has not shown that it lists the right tree (review of
    PR 495)."""
    names = set(names)
    if "DECOY_MARK" in names:
        return "decoy"
    return "original" if "data.json" in names else "empty"


def classify_scan(seen) -> str:
    """The verdict for a scan: the same rule as `classify_listing`, on the parts of each entry the scan reported."""
    seen = {tuple(p) for p in seen}
    if any(p and p[-1] == "DECOY_MARK" for p in seen):
        return "decoy"
    return "original" if ("sub", "data.json") in seen else "empty"


def _r_list_beneath(pin, env):
    from orchard import safeio
    with safeio.list_beneath(pin.fd, ("sub",)) as it:
        names = sorted(e.name for e in it)
    return classify_listing(names)


def _r_scan_beneath(pin, env):
    from orchard import safeio
    seen: list[tuple[str, ...]] = []
    safeio.scan_beneath(pin, lambda entry: seen.append(tuple(entry.parts)))       # ScanEntry has `parts` and `kind`
    return classify_scan(seen)                                                  # the real tree must be listed, or the scan proved nothing


def published_regular(path: Path, data: bytes) -> bool:
    """True when `path` is a regular file (not a link) holding exactly `data`. A write that says it succeeded must pass this."""
    try:
        st = os.lstat(path)
        return stat.S_ISREG(st.st_mode) and path.read_bytes() == data
    except OSError:
        return False


def published_beneath(root_fd: int, parts, data: bytes) -> bool:
    """True when the regular file `parts` beneath the pinned directory holds exactly `data`, read through the descriptor."""
    from orchard import safeio
    try:
        return safeio.read_beneath(root_fd, parts, max_bytes=1 << 20) == data
    except (OSError, safeio.SafeReadError):                 # a refusal or a missing file is "not published"; a timeout is not caught here
        return False


def _r_write_beneath(pin, env):
    from orchard import safeio
    safeio.write_beneath(pin.fd, ("sub", "written.txt"), b"w")
    ok = published_beneath(pin.fd, ("sub", "written.txt"), b"w")                            # checked now, while the pin is still open
    env["publishes"] = lambda: ok
    return "written"


def _r_read_bytes_path(env):
    from orchard import safeio
    return _label(safeio.read_bytes(env["file"], max_bytes=1 << 20))


def _r_read_json_path(env):
    from orchard import safeio
    return _json_label(safeio.read_json(env["file"], max_bytes=1 << 20))


def _r_drop_check_dir_client(env):
    from orchard import drop_check
    return _label(drop_check.DirClient(env["file"].parent).get("Qwen/x", "0" * 40, env["file"].name, max_bytes=1 << 20))


def _r_atomic_write_path(env):
    from orchard import safeio
    safeio.atomic_write(env["root"] / "sub" / "data.json", b"new")
    env["publishes"] = lambda: published_regular(env["root"] / "sub" / "data.json", b"new")
    return "written"


def _r_ledger_lock(env):
    from orchard import ledger
    led = ledger.Ledger(env["root"] / "ledger.jsonl")
    try:
        # The lock file the reader opened must be the one the case prepared; if the name drifted the case proves nothing.
        return "original" if env["lock"].exists() else "other"
    finally:
        led.close()


def _r_cache_lock(env):
    from orchard import cache_lock
    with cache_lock.entry_lock(env["root"], "ledger.jsonl", exclusive=True) as held:
        return "original" if held and env["cache_lock"].exists() else ("refused" if not held else "other")


def _r_residency_headers(env):
    from orchard import residency
    got = residency.read_headers(env["snapshot"])
    return "original" if [t.name for t in got.tensors] == ["w"] else "other"


def sidecar_leaked(entries: list[dict]) -> bool:
    """True when a listed `extra.gguf` carries the size or hash of the file the link points at. Listing the link by name, with a reason
    and no target size, is the correct answer (issue 464) and is not a leak."""
    return any(e["file"] == "extra.gguf" and (e.get("size") == len(VICTIM_BYTES) or bool(e.get("sha256"))) for e in entries)


def classify_sidecars(entries: list[dict], *, expect_listed: str | None) -> str:
    """The verdict for a sidecar scan: `decoy` when an entry carries the link target's size or hash, `empty` when the scan was expected to
    list `expect_listed` (the link is listed by name) and did not, otherwise `original`."""
    if sidecar_leaked(entries):
        return "decoy"
    if expect_listed is not None and not any(e.get("file") == expect_listed for e in entries):
        return "empty"
    return "original"


def _r_compare_sidecars(env):
    import importlib.util
    path = REPO / "orchard" / "skills" / "delta-triage-templates" / "delta_triage.py"
    spec = importlib.util.spec_from_file_location("delta_triage_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    empty = env["base"] / "empty_base"
    empty.mkdir()
    got = mod.compare_sidecars(env["root"], empty, set())
    # A refused link may stay listed (issue 464: with a reason and no target size). It leaks when an entry carries the target's size or hash.
    # With the link in the tree the scan must name it, or an empty result would pass for a scan that refused to look.
    return classify_sidecars(got["sidecars"] + got["ignored_files"], expect_listed="extra.gguf" if env["case"] == "sidecar_link" else None)


def _r_triage_capabilities(env):
    """The stage 0 template's reader of a bundle's orchard-capabilities.json, pointed at data.json. A named problem is a refusal."""
    import importlib.util
    from orchard import safeio
    path = REPO / "orchard" / "skills" / "delta-triage-templates" / "delta_triage.py"
    spec = importlib.util.spec_from_file_location("delta_triage_caps_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.CAPABILITIES_FILE = env["file"].name
    rec, data = mod.read_capabilities(str(env["file"].parent))
    if rec["problems"] or data is None:
        raise safeio.ReadFailed("; ".join(rec["problems"]) or "absent")
    return _json_label(data)


def _r_bundle_capabilities(env):
    """The stage 0 gate's hash of a bundle's orchard-capabilities.json (orchard/bundle_capabilities.py), pointed at data.json."""
    import hashlib
    from orchard import bundle_capabilities as bc, safeio
    old, bc.FILE = bc.FILE, env["file"].name
    try:
        sha, problem = bc.file_state(env["file"].parent)
    finally:
        bc.FILE = old
    if problem or sha is None:
        raise safeio.ReadFailed(problem or "absent")
    return {hashlib.sha256(OK_BYTES).hexdigest(): "original", hashlib.sha256(DECOY_BYTES).hexdigest(): "decoy"}.get(sha, "other")


def _r_port_sources(env):
    """The port generator's walk of a port directory (orchard/port_template.py port_sources, behind check_port), pointed at the tree's
    .json files so the verdict comes from what it listed and read."""
    from orchard import port_template
    seen = port_template.port_sources(env["root"], suffix=".json")
    return classify_scan([tuple(rel.split("/")) for rel in seen])


PATH_READERS: list[PathReader] = [
    PathReader("safeio.read_bytes(anchor=pin)", "pinned_read", _pinned(_r_read_bytes_anchor)),
    PathReader("port_template.port_sources", "scan", _r_port_sources, claims_pin=False),
    PathReader("safeio.read_beneath", "pinned_read", _pinned(_r_read_beneath)),
    PathReader("Pin.read_json", "pinned_read", _pinned(_r_read_json_pin)),
    PathReader("safeio.sha256_file(anchor=pin)", "pinned_read", _pinned(_r_sha256_anchor)),
    PathReader("safeio.list_beneath", "pinned_list", _pinned(_r_list_beneath)),
    PathReader("safeio.scan_beneath", "scan", _pinned(_r_scan_beneath)),
    PathReader("safeio.write_beneath", "pinned_write", _pinned(_r_write_beneath)),
    PathReader("safeio.read_bytes(path)", "path_read", _r_read_bytes_path, claims_pin=False),
    PathReader("safeio.read_json(path)", "path_read", _r_read_json_path, claims_pin=False),
    PathReader("drop_check.DirClient.get", "path_read", _r_drop_check_dir_client, claims_pin=False),
    PathReader("safeio.atomic_write(path)", "path_write", _r_atomic_write_path, claims_pin=False),
    PathReader("ledger.Ledger lock file", "lock_open", _r_ledger_lock, claims_pin=False),
    PathReader("cache_lock.entry_lock", "lock_open", _r_cache_lock, claims_pin=False),
    PathReader("residency.read_headers", "cache_headers", _r_residency_headers, claims_pin=False),
    PathReader("delta_triage.compare_sidecars", "sidecar_scan", _r_compare_sidecars, claims_pin=False),
    PathReader("delta_triage.read_capabilities", "path_read", _r_triage_capabilities, claims_pin=False),
    PathReader("bundle_capabilities.file_state", "path_read", _r_bundle_capabilities, claims_pin=False),
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


def _c_scan_by_name(env):
    from orchard import safeio
    with safeio.pinned_root(env["root"]):
        env["between"]()
        seen = [Path(d) / n for d, _, names in os.walk(env["root"]) for n in names]       # mistake 2 for a tree: walk by name after the pin
    return "decoy" if any(p.name == "DECOY_MARK" for p in seen) else "original"


def _c_deletes_victim(env):
    env["victim"].unlink()                                    # mistake: removes a file outside the root, then says nothing happened
    return "original"


def _c_deletes_outside_dir(env):
    import shutil
    shutil.rmtree(env["outside"] / "sub")                     # mistake: removes a directory outside the root
    return "original"


def _c_replaces_with_link(env):
    data = env["outside"] / "sub" / "data.json"
    copy = env["outside"] / "copy.json"
    copy.write_bytes(data.read_bytes())
    data.unlink()
    data.symlink_to(copy)                                     # mistake: the bytes read back are the same, but the file is now a link
    return "refused"


def _c_replaces_dir_with_link(env):
    import shutil
    sub = env["outside"] / "sub"
    shutil.copytree(sub, env["outside"] / "sub_copy")
    shutil.rmtree(sub)
    sub.symlink_to(env["outside"] / "sub_copy")               # mistake: same names and bytes, but the directory is now a link
    return "refused"


def _c_writes_nothing(env):
    env["publishes"] = lambda: published_regular(env["root"] / "sub" / "never.txt", b"w")
    return "written"                                          # mistake: claims success, publishes nothing


def _c_writes_without_check(env):
    return "written"                                          # mistake: claims success and supplies no publication check


def _c_rewrites_decoy_mark(env):
    (env["outside"] / "sub" / "DECOY_MARK").write_bytes(b"changed")      # mistake: a fixture file nobody compared before
    return "original"


def _c_adds_outside_file(env):
    (env["outside"] / "new.txt").write_bytes(b"x")                       # mistake: a new file outside the root
    return "original"


def _c_lists_nothing(pin, env):
    return classify_listing([])                                          # mistake: an empty listing reported as a pass


def _c_scans_nothing(pin, env):
    return classify_scan([])


def _c_swallows_the_alarm(env):
    try:
        with open(env["file"], "rb") as fh:                              # blocks on the FIFO
            return _label(fh.read())
    except Exception:                                                    # mistake: a blanket handler turns the timeout into a refusal
        return "refused"


CONTROLS: list[PathReader] = [
    PathReader("control: rewrites a fixture file outside the root", "effects", _c_rewrites_decoy_mark),
    PathReader("control: adds a file outside the root", "effects", _c_adds_outside_file),
    PathReader("control: lists nothing", "pinned_list", _pinned(_c_lists_nothing)),
    PathReader("control: scans nothing", "scan", _pinned(_c_scans_nothing)),
    PathReader("control: swallows the alarm", "path_read", _c_swallows_the_alarm, claims_pin=False),
    PathReader("control: deletes the file outside the root", "effects", _c_deletes_victim),
    PathReader("control: deletes the directory outside the root", "effects", _c_deletes_outside_dir),
    PathReader("control: replaces the file outside the root with a link", "effects", _c_replaces_with_link),
    PathReader("control: replaces the directory outside the root with a link", "effects", _c_replaces_dir_with_link),
    PathReader("control: write that publishes nothing", "effects", _c_writes_nothing),
    PathReader("control: write that omits the publication check", "effects", _c_writes_without_check),
    PathReader("control: pin, then walk by pathname", "scan", _c_scan_by_name),
    PathReader("control: builtin open", "path_read", _c_naive_open, claims_pin=False),
    PathReader("control: O_NOFOLLOW on the last part only", "pinned_read", _c_nofollow_last_only),
    PathReader("control: pin, then read by pathname", "pinned_read", _c_path_after_pin),
]

# family -> case -> the verdicts that are right. Anything else is a failure of the reader.
READ_BAD = {"leaked", "hang", "crash"}
EXPECT: dict[str, dict[str, set[str]]] = {
    "effects": {"plain": {"original", "written"}, "ancestor_link": {"original", "refused", "written"}},      # only the controls use this family
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

def snapshot_tree(top: Path) -> dict:
    """Every entry under `top`, by relative path: a regular file as its bytes, a link as its target text, a directory as `dir`, anything
    else (a FIFO, a device) as its kind. Nothing is followed and no special file is opened, so taking it cannot block."""
    out: dict[str, object] = {}

    def walk(directory: Path, prefix: str) -> None:
        for entry in sorted(os.scandir(directory), key=lambda e: e.name):
            rel = prefix + entry.name
            mode = entry.stat(follow_symlinks=False).st_mode
            if stat.S_ISLNK(mode):
                out[rel] = ("link", os.readlink(entry.path))
            elif stat.S_ISDIR(mode):
                out[rel] = ("dir",)
                walk(Path(entry.path), rel + "/")
            elif stat.S_ISREG(mode):
                out[rel] = ("file", Path(entry.path).read_bytes())
            else:
                out[rel] = ("special", stat.S_IFMT(mode))
    if top.is_dir() and not top.is_symlink():
        walk(top, "")
    return out


def outside_changes(before: dict, after: dict) -> list[str]:
    """The paths under the outside directory that were added, removed or changed between two snapshots."""
    return sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))


class _Alarm(BaseException):
    """The per-case alarm. It derives from BaseException so that no `except Exception` in a reader, a helper or the harness can swallow a
    timeout and turn a blocked FIFO read into a quiet refusal (review of PR 495)."""


def _verdict(reader: PathReader, case_name: str, base: Path) -> dict:
    env = layout(base)
    env["case"] = case_name
    CASES[case_name].build(env)
    # Every file the fixture put outside the root is compared before and after, so a reader that changes `DECOY_MARK`, the
    # Hugging Face decoy weights or adds a file must be seen (review of PR 495).
    outside_before = snapshot_tree(env["outside"])

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
    # Effects outside the root count as a leak whatever the reader returned. An object that existed before must still exist, as the same
    # kind of object holding the same bytes: a reader that deletes or replaces it must not pass because there is nothing left to compare.
    publishes = env.get("publishes")
    changed = outside_changes(outside_before, snapshot_tree(env["outside"]))
    if changed:
        got, detail = "leaked", f"a path outside the root changed, was replaced, added or deleted: {', '.join(changed[:4])}"
    elif got == "decoy":
        got, detail = "leaked", "the reader returned content from outside the root"
    elif got == "written" and (publishes is None or not publishes()):
        # A writer that reports success without supplying a publication check cannot be verified, so it counts as unpublished.
        got, detail = "unpublished", "the writer reported success but the destination does not hold the bytes it wrote (or gave no way to check)"
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
    try:
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:                      # the child's own alarms did not fire: that is a hang, reported as one
        return [{"case": "child", "verdict": "hang", "detail": f"the child did not finish in {timeout} s"}]
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
