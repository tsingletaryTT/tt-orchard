# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""One corpus of hostile inputs for every reader of a file or record that something else wrote.

Two users:
  * `tests/test_safeio.py` feeds the corpus to `orchard/safeio.py` and proves each guard.
  * `tests/test_hostile_readers.py` feeds the same corpus to every reader registered in `READERS`
    (the registry `READERS` is at the end of this module) and states what each reader must do.

An item builds one hostile thing at a path the caller chooses: a file with hostile content, a directory, a FIFO, a link to a
regular file, a link to /dev/zero, a file nobody may read. Items that need a sequence of actions between a check and an
open (a directory swapped for a link, a file that grows while it is read) are not in the corpus. They are tests of the
reader's open path and live in test_safeio.py, where the hook that makes the swap is installed.
"""
from __future__ import annotations

import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# --------------------------------------------------------------------------------------------------------- byte payloads

BIG_STRING_BYTES = 50 * 1024 * 1024          # the size of the large string item


def _deep(n: int) -> bytes:
    return b"[" * n + b"]" * n


def _deep_object(n: int) -> bytes:
    return b'{"a":' * n + b"1" + b"}" * n


# Name -> content. Each is something a writer could leave in a JSON file, accidentally or on purpose. Some are valid JSON of an
# unexpected type or value ([], {}, null, 0, -1, true, "x", ""), some are not JSON at all, some are JSON that Python accepts and
# a reader must still refuse (NaN, Infinity, 1e999, a repeated key, a lone surrogate).
BYTES_CORPUS: dict[str, bytes] = {
    "empty_array": b"[]",
    "empty_object": b"{}",
    "null": b"null",
    "empty_string": b'""',
    "string_x": b'"x"',
    "zero": b"0",
    "minus_one": b"-1",
    "huge_exponent": b"1e999",
    "nan": b"NaN",
    "infinity": b"Infinity",
    "negative_infinity": b"-Infinity",
    "nan_in_object": b'{"a": NaN, "b": [Infinity]}',
    "true": b"true",
    "nest_1000": _deep(1000),
    "nest_1000_object": _deep_object(1000),
    "nest_100000": _deep(100_000),
    "duplicate_keys": b'{"a": 1, "a": 2}',
    "lone_surrogate": b'"\\ud800"',
    "lone_surrogate_key": b'{"\\udc00": 1}',
    "invalid_utf8": b'"\xff\xfe"',
    "bom": b"\xef\xbb\xbf{}",
    "truncated": b'{"a": [1, 2',
    "empty": b"",
    "whitespace_only": b" \n\t",
    "trailing_garbage": b"{} x",
    "huge_integer": b"1" + b"0" * 5000,
    "null_byte_in_string": b'{"a": "\x00"}',
    "negative_zero_count": b'{"count": -0.0}',
    "float_count": b'{"count": 5.0}',
    "negative_count": b'{"count": -3}',
    "string_count": b'{"count": "5"}',
    "bool_count": b'{"count": true}',
    "negative_long_integer": b"-" + b"1" * 100,
    "unhashable_dtype": b'{"a": {"dtype": [], "shape": [], "data_offsets": []}}',
}


def big_string() -> bytes:
    """A JSON string of 50 MB. Built on demand; held by no module-level name."""
    return b'"' + b"a" * (BIG_STRING_BYTES - 2) + b'"'


# --------------------------------------------------------------------------------------------------------------- items

@dataclass(frozen=True)
class Item:
    """One hostile thing. `build(dest)` creates it at the path `dest` (its parent exists) and may leave other files beside it."""
    name: str
    build: Callable[[Path], None]
    kind: str                        # "bytes" (a regular file with these bytes), "special" (not a regular file), "denied", "big"
    content: bytes | None = None     # for kind "bytes"


def _write_bytes(data: bytes) -> Callable[[Path], None]:
    def build(dest: Path) -> None:
        dest.write_bytes(data)
    return build


def _directory(dest: Path) -> None:
    dest.mkdir()
    (dest / "inner.json").write_text("{}")


def _fifo(dest: Path) -> None:
    os.mkfifo(dest)


def _link_to_regular(dest: Path) -> None:
    target = dest.with_name(dest.name + ".real")
    target.write_text("{}")
    dest.symlink_to(target.name)


def _link_to_dev_zero(dest: Path) -> None:
    dest.symlink_to("/dev/zero")


def _dangling_link(dest: Path) -> None:
    dest.symlink_to("no-such-file-anywhere")


def _link_loop(dest: Path) -> None:
    dest.symlink_to(dest.name)


def _denied(dest: Path) -> None:
    dest.write_text("{}")
    dest.chmod(0)


def _big_string_file(dest: Path) -> None:
    with open(dest, "wb") as f:
        f.write(b'"')
        chunk = b"a" * (1 << 20)
        for _ in range(BIG_STRING_BYTES // (1 << 20) - 1):
            f.write(chunk)
        f.write(b'"')


def running_as_root() -> bool:
    return os.geteuid() == 0


ITEMS: list[Item] = (
    [Item(name, _write_bytes(data), "bytes", data) for name, data in BYTES_CORPUS.items()]
    + [
        Item("directory", _directory, "special"),
        Item("fifo", _fifo, "special"),
        Item("link_to_regular_file", _link_to_regular, "special"),
        Item("link_to_dev_zero", _link_to_dev_zero, "special"),
        Item("dangling_link", _dangling_link, "special"),
        Item("link_loop", _link_loop, "special"),
        Item("permission_denied", _denied, "denied"),
        Item("big_string_50mb", _big_string_file, "big"),
    ]
)

# Items that can make a reader read without end or hold a lot of memory. They run in a child process with an address-space limit.
HEAVY = frozenset({"link_to_dev_zero", "big_string_50mb"})


def make_item(item: Item, dest: Path) -> None:
    """Build `item` at `dest`. A permission-denied item is built as an ordinary file when the process is root, since root is
    never denied; the test that needs the denial skips then."""
    item.build(dest)


def restore_permissions(root: Path) -> None:
    """Give every file under `root` read and write permission again so a temporary directory can be removed."""
    for path in Path(root).rglob("*"):
        try:
            if not path.is_symlink() and not stat.S_ISFIFO(path.lstat().st_mode):
                path.chmod(0o700 if path.is_dir() else 0o600)
        except OSError:
            pass


# -------------------------------------------------------------------------------------------------------- child process

REPO = Path(__file__).resolve().parent.parent


def run_limited(code: str, *, timeout: float = 60, address_space: int = 2 << 30):
    """Run Python `code` in a child with an address-space limit and a timeout (the repo's rule for a hostile-input test: a reader
    that reads /dev/zero to the end would exhaust memory, and the limit turns that into a MemoryError in the child alone).
    The repository root is on the child's sys.path. Returns the CompletedProcess; a timeout raises subprocess.TimeoutExpired."""
    import subprocess
    import sys
    prologue = (f"import resource, sys; resource.setrlimit(resource.RLIMIT_AS, ({address_space}, {address_space})); "
                f"sys.path.insert(0, {str(REPO)!r})\n")
    return subprocess.run([sys.executable, "-c", prologue + code], capture_output=True, text=True, timeout=timeout)










# ------------------------------------------------------------------------------------------------------------ the registry

@dataclass(frozen=True)
class Reader:
    """One reader of a file or record that something else wrote, and what it must do with every hostile input.

    `call(arg)` runs the reader. For `feed == "file"`, `arg` is a fresh base directory, `setup(base)` has been run, and the hostile
    item was built at `base / rel`. For `feed == "bytes"`, `arg` is the payload (the reader gets it from a fake fetcher or as text).
    The contract the test enforces on every item:

      * it returns, or raises one of `documented` (an exception class that is part of the reader's contract), and nothing else;
      * what it returns holds no NaN or Infinity, no lone surrogate and no nesting deeper than 50, and `check(result)` (optional)
        returns None;
      * it finishes within a few seconds (a FIFO must not block it);
      * on the 50 MB string it does not read most of the file, and on a link to /dev/zero it does not read without end.

    `os_error_ok` lists the items for which a bare OSError counts as documented (a reader cannot invent file permissions).
    A reader that fails an item today is listed in `KNOWN_DEFECTS` (below) with the GitHub issue that tracks the defect."""
    name: str
    call: Callable[[object], object]
    feed: str = "file"
    rel: str = "input.json"
    suffix: bytes = b""           # appended to a bytes item: a line reader needs the newline that ends the hostile line
    setup: Callable[[Path], None] | None = None
    documented: tuple = ()
    check: Callable[[object], str | None] | None = None
    os_error_ok: frozenset = frozenset({"permission_denied", "dangling_link", "link_loop"})


MAX_RETURNED_DEPTH = 50
PEAK_LIMIT_MB = 30                # reading a 50 MB file into memory shows up as at least this much traced memory
ITEM_TIMEOUT_S = {"fifo": 3, "link_to_dev_zero": 25, "big_string_50mb": 25}
DEFAULT_TIMEOUT_S = 5


def result_problem(value) -> str | None:
    """A defect in what a reader returned, whatever the reader is: NaN or Infinity, a lone surrogate, nesting deeper than 50."""
    import dataclasses
    import math
    stack = [(value, 1)]
    seen = 0
    while stack:
        item, depth = stack.pop()
        seen += 1
        if seen > 200_000:
            return "returned a structure with more than 200,000 parts"
        if depth > MAX_RETURNED_DEPTH:
            return f"returned a structure nested more than {MAX_RETURNED_DEPTH} deep"
        if isinstance(item, float) and not math.isfinite(item):
            return "returned NaN or Infinity"
        if isinstance(item, str) and any(0xD800 <= ord(c) <= 0xDFFF for c in item):
            return "returned text with a lone surrogate"
        if isinstance(item, dict):
            stack.extend((k, depth + 1) for k in item)
            stack.extend((v, depth + 1) for v in item.values())
        elif isinstance(item, (list, tuple, set, frozenset)):
            stack.extend((v, depth + 1) for v in item)
        elif dataclasses.is_dataclass(item) and not isinstance(item, type):
            stack.extend((getattr(item, f.name), depth + 1) for f in dataclasses.fields(item))
    return None


def _verdict(reader: Reader, item: Item, arg) -> dict:
    """Run `reader` on `arg` with a timer and say whether the contract held. Runs in the child process."""
    import signal

    class Timeout(BaseException):
        pass

    def on_alarm(signum, frame):
        raise Timeout()

    seconds = ITEM_TIMEOUT_S.get(item.name, DEFAULT_TIMEOUT_S)
    old = signal.signal(signal.SIGALRM, on_alarm)
    import tracemalloc
    measure = item.name == "big_string_50mb" and reader.feed == "file"   # traced memory is judged only for readers that open the file themselves; the peak resident size of a child is inherited from its parent, so it cannot be used
    if measure:
        tracemalloc.start()
    outcome: dict
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        try:
            result = reader.call(arg)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
    except Timeout:
        outcome = {"ok": False, "why": f"did not finish within {seconds} s"}
    except MemoryError:
        outcome = {"ok": False, "why": "ran out of memory (reads without a bound)"}
    except RecursionError as exc:
        outcome = {"ok": False, "why": "raised RecursionError, which is not documented"}
    except Exception as exc:                                   # a hostile input may raise anything
        allowed = isinstance(exc, reader.documented) or (isinstance(exc, OSError) and item.name in reader.os_error_ok)
        outcome = {"ok": allowed, "why": "" if allowed else f"raised {type(exc).__module__}.{type(exc).__qualname__}, which is not documented"}
    else:
        problem = result_problem(result) or (reader.check(result) if reader.check else None)
        outcome = {"ok": problem is None, "why": problem or ""}
    finally:
        signal.signal(signal.SIGALRM, old)
    peak = 0.0
    if measure:
        peak = tracemalloc.get_traced_memory()[1] / 1e6
        tracemalloc.stop()
    if measure and outcome["ok"] and peak > PEAK_LIMIT_MB:
        outcome = {"ok": False, "why": f"held {peak:.0f} MB in memory for a 50 MB file (it reads past any cap)"}
    outcome["peak_mb"] = round(peak, 1)
    return outcome


def child_main(reader_name: str, group: str, registry: dict | None = None) -> None:
    """Entry point of the child process: run one reader against the light items, or against one heavy item, and print one JSON
    line `{item: {ok, why, peak_mb}}`. `group` is "light" or the name of a heavy item."""
    import json
    import shutil
    import tempfile
    readers = registry if registry is not None else {**BY_NAME, **SELFTEST}
    reader = readers[reader_name]
    items = [i for i in ITEMS if (i.name == group if group != "light" else i.name not in HEAVY)]
    out = {}
    for item in items:
        if reader.feed == "bytes" and item.kind not in ("bytes", "big"):
            continue
        base = Path(tempfile.mkdtemp(prefix="hostile-"))
        try:
            if reader.feed == "file":
                if reader.setup:
                    reader.setup(base)
                dest = base / reader.rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                item.build(dest)
                if reader.suffix and item.kind == "bytes":
                    with open(dest, "ab") as f:
                        f.write(reader.suffix)
                arg = base
            else:
                arg = big_string() if item.kind == "big" else item.content
            out[item.name] = _verdict(reader, item, arg)
        except BaseException as exc:                           # building the item failed: say so instead of passing
            out[item.name] = {"ok": False, "why": f"the harness could not build the item: {type(exc).__name__}", "peak_mb": 0}
        finally:
            restore_permissions(base)
            shutil.rmtree(base, ignore_errors=True)
    print(json.dumps(out))


def run_reader(reader: Reader, *, module: str = "hostile_inputs") -> dict:
    """{item name: {ok, why, peak_mb}} for one reader, from child processes with a 2 GiB address-space limit and a timeout.
    The light items share one child; each heavy item has its own, so a reader that exhausts memory on one does not hide the other."""
    import json
    import subprocess
    import sys
    results: dict = {}
    groups = ["light"] + sorted(HEAVY)
    for group in groups:
        code = (f"import sys; sys.path[:0] = [{str(REPO)!r}, {str(Path(__file__).resolve().parent)!r}]\n"
                f"import {module} as m\nm.child_main({reader.name!r}, {group!r})\n")
        try:
            done = run_limited(code, timeout=90)
        except subprocess.TimeoutExpired:
            results.update({n: {"ok": False, "why": "the child process did not finish in 90 s"} for n in _group_items(reader, group)})
            continue
        lines = [ln for ln in done.stdout.splitlines() if ln.startswith("{")]
        if done.returncode != 0 or not lines:
            why = f"the child process failed (exit {done.returncode}): {done.stderr.strip()[-300:]}"
            results.update({n: {"ok": False, "why": why} for n in _group_items(reader, group)})
            continue
        results.update(json.loads(lines[-1]))
    return results


def _group_items(reader: Reader, group: str) -> list[str]:
    return [i.name for i in ITEMS if (i.name == group if group != "light" else i.name not in HEAVY)
            and not (reader.feed == "bytes" and i.kind not in ("bytes", "big"))]


# ----------------------------------------------------------------------------------------------------------- the readers

def _valid_ledger(base: Path) -> None:
    """A run directory with a real, valid ledger, so a hostile item elsewhere in the directory is the only fault."""
    from orchard.ledger import Ledger
    with Ledger(base / "ledger.jsonl") as ledger:
        ledger.append("run_start", None, model="org/model")


def _collect(base: Path):
    from orchard import status
    return status.collect(base, now=1000.0, pid_alive=lambda pid: False, lock_holder=lambda run: None,
                          disk_free_gb=lambda path: 1.0, gozer_status=lambda: "", home=base)


def _ledger_read_entries(base: Path):
    from orchard import ledger
    return ledger.read_entries(base / "input.json")


def _telegraph_store_read_snapshot(base: Path):
    """The ledger database is the hostile item: a file, link, FIFO, directory or damaged file where `project.db` should be."""
    from orchard import telegraph_store
    return telegraph_store.read_snapshot(base)


def _telegraph_migrate_read_archive_lines(base: Path):
    from orchard import telegraph_migrate
    return telegraph_migrate.read_archive_lines(base)


def _telegraph_migrate_dry_run(base: Path):
    """The importer's check of an old jsonl ledger; it writes nothing."""
    from orchard import telegraph_migrate
    return telegraph_migrate.migrate(base, dry_run=True)


def _stages_load(base: Path):
    from orchard import stages
    return stages._load(base, "input.json")


def _stages_evidence_json(base: Path):
    from orchard import stages
    reasons: list = []
    data = stages._evidence_json(base, "input.json", required=True, reasons=reasons)
    return data, reasons


def _stages_hw_record(base: Path):
    from orchard import stages
    return stages.hw_record_problem(base, 2)


def _stages_suite_reasons_in(base: Path):
    from orchard import stages
    return stages._suite_reasons_in(base / "input.json", base)


def _stages_read_json(base: Path):
    from orchard import stages
    return stages._read_json(base / "input.json")


def _stages_gate_delta(base: Path):
    from orchard import stages
    (base / "run").mkdir(exist_ok=True)
    return stages.gate_delta(base / "stage", base / "run")


def _package_read_reported(base: Path):
    from orchard import package
    return package.read_reported(base)


def _package_json(base: Path):
    from orchard import package
    return package._json(base / "input.json", "a test file")


def _package_stage_2_vocab(base: Path):
    from orchard import package
    return package.stage_2_vocab(base)


def _package_graft_record(base: Path):
    from orchard import package
    return package._stage_2_graft_record(base)


def _audio_package_read_json_file(base: Path):
    from orchard import audio_package
    return audio_package.read_json_file(base / "input.json", "a test file")


def _audio_run(base: Path, *, ledger: bool = True) -> None:
    """A valid audio run in `base` (ledger and result), so the hostile item is the only thing wrong."""
    import json
    import test_audio_package as tap
    (base / "stages" / "2").mkdir(parents=True, exist_ok=True)
    if ledger:
        from orchard.ledger import Ledger
        with Ledger(base / "ledger.jsonl") as led:
            led.append("run_start", model="stabilityai/stable-audio-3-small-music")
            led.append("decision", decision="ready for operator review")
    (base / "stages" / "2" / "audio_result.json").write_text(json.dumps(tap.audio_result()))
    (base / "bench.json").write_text(json.dumps(tap.bench_file()))


def _audio_package_read_facts_result(base: Path):
    from orchard import audio_package
    (base / "stages" / "2").mkdir(parents=True, exist_ok=True)
    (base / "ledger.jsonl").write_text("{}\n")
    (base / "bench.json").write_text("{}")
    return audio_package.read_facts(base, base / "bench.json")


def _audio_package_read_facts_bench(base: Path):
    from orchard import audio_package
    _audio_run(base)
    return audio_package.read_facts(base, base / "input.json")


def _audio_package_read_facts_ledger(base: Path):
    from orchard import audio_package
    _audio_run(base, ledger=False)
    return audio_package.read_facts(base, base / "bench.json")


def _bench_rerun_read_samples(base: Path):
    from orchard import bench_rerun
    return bench_rerun.read_samples(base, "phase")


def _opinventory_load_table(base: Path):
    from orchard import opinventory
    return opinventory.load_table(base / "input.json")


def _opinventory_load_ignored(base: Path):
    from orchard import opinventory
    return opinventory.load_ignored(base / "input.json")


def _opinventory_load_dump(base: Path):
    from orchard import opinventory
    return opinventory.load_dump("2.0.0", data_dir=base)


def _opinventory_load_verified(base: Path):
    from orchard import opinventory
    return opinventory.load_verified(base / "input.json")


def _opvariants_load_data(base: Path):
    from orchard import opvariants
    return opvariants.load_data.__wrapped__(base / "input.json")


def _verdict_read_json(base: Path):
    from orchard import verdict
    return verdict._read_json(base, "input.json")


def _sampling_choice_read_json(base: Path):
    from orchard import sampling_choice
    return sampling_choice._read_json(base / "input.json")


def _kept_coder_read(base: Path):
    from orchard import kept_coder
    return kept_coder.read(base)


def _speedwatch_baselines(base: Path):
    from orchard import speedwatch
    return speedwatch.Baselines(base / "input.json")._read()


def _run_view_read(base: Path):
    from orchard import run_view
    return run_view._read(base / "input.json")


def _prompt_suite_load(base: Path):
    from orchard import prompt_suite
    return prompt_suite.load_suite(base / "input.json")


def _supervisor_container_config(base: Path):
    from orchard.supervisor import Supervisor
    return Supervisor._container_config(base)


def _supervisor_swap_cache(base: Path):
    from orchard.supervisor import Supervisor
    return Supervisor._swap_cache(base)


def _card_use_tool_call_evidence(base: Path):
    from orchard import card_use
    return card_use.tool_call_evidence(base, ("org/model", None))


def _transcripts_load_signature(base: Path):
    from orchard import transcripts
    return transcripts.load_signature(base / "input.json")


def _kernelprobe_read_agent_check(base: Path):
    from orchard import kernelprobe
    return kernelprobe._read_agent_check(base / "input.json")


def _skillsfresh_cache(base: Path):
    from orchard import skillsfresh
    return skillsfresh.Cache(base / "input.json", write=False).data


def _toolcall_bench_rows(base: Path):
    from orchard import toolcall_bench
    return toolcall_bench._rows(base / "input.json")


def _stock_runtime_load_config(base: Path):
    from orchard import stock_runtime
    return stock_runtime._load_config(base / "input.json")


def _model_config_read(base: Path):
    from orchard import model_config
    return model_config.read_model_config(base / "input.json")


def _model_config_dims(base: Path):
    """The whole path stage 0 would take: read the file, then normalise it. Both steps raise ModelConfigError and nothing else."""
    from orchard import model_config
    return model_config.dims(model_config.read_model_config(base / "input.json")).__dict__


def _swap_draft_installed_containers(base: Path):
    from orchard import swap_draft
    return swap_draft.installed_containers(base)


def _nearest_read_containers(base: Path):
    from orchard import nearest
    return nearest.read_containers(base)


def _classify_bench_load_corpus(base: Path):
    from orchard import classify_bench
    return classify_bench.load_corpus(base / "input.json")


def _bench_session_load_items(base: Path):
    from orchard import bench_session
    return bench_session.load_items(base / "input.json")


class _FakeFetch:
    """The fetcher read_model takes, serving `payload` where the test says. It honours byte_range and max_bytes as the real one does."""

    def __init__(self, routes: dict):
        self.routes = routes

    def get(self, url, *, byte_range=None, max_bytes=8 * 1024 * 1024):
        from orchard.research.http import FetchError
        data = self.routes.get(url.rsplit("/", 1)[-1])
        if data is None:
            raise FetchError("HTTP 404")
        if byte_range is not None:
            data = data[byte_range[0]:byte_range[1] + 1]
        return data[:max_bytes]

    def json(self, url, **kw):
        """The real Fetcher.json over this fake's `get`, so the strict parse that read_model gets from the real fetcher is the one tested."""
        from orchard.research.http import Fetcher
        return Fetcher.json(self, url, **kw)


def _header_file(payload: bytes) -> bytes:
    import struct
    return struct.pack("<Q", len(payload)) + payload


def _headers_read_model_header(payload: bytes):
    from orchard.research import headers
    fetch = _FakeFetch({"model.safetensors": _header_file(payload)})
    return headers.read_model(fetch, "org/model", ["model.safetensors"])


def _headers_read_model_index(payload: bytes):
    from orchard.research import headers
    good = _header_file(b'{"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}')
    fetch = _FakeFetch({"model.safetensors.index.json": payload, "model-00001-of-00002.safetensors": good,
                        "model-00002-of-00002.safetensors": good})
    return headers.read_model(fetch, "org/model", ["model.safetensors.index.json", "model-00001-of-00002.safetensors",
                                                    "model-00002-of-00002.safetensors"])


def _headers_shard_header(payload: bytes):
    from orchard.research import headers
    return headers.shard_header(_FakeFetch({"m.safetensors": _header_file(payload)}), "org/model", "m.safetensors")


def _http_fetcher_json(payload: bytes):
    from orchard.research.http import Fetcher
    fetcher = object.__new__(Fetcher)
    fetcher.get = lambda url, **kw: payload
    return fetcher.json("https://huggingface.co/org/model/resolve/main/config.json")


def _types():
    """The exception classes named in the registry. Imported here so that importing this module stays cheap."""
    from orchard.ledger import LedgerCorrupt
    from orchard.package import PackageError
    from orchard.prompt_suite import SuiteError
    from orchard.research.http import FetchError
    from orchard.db import LedgerRefused, LedgerUnavailable
    from orchard.classify_bench import CorpusError
    from orchard.safeio import SafeReadError
    from orchard.telegraph_migrate import MigrationRefused
    return LedgerCorrupt, PackageError, SuiteError, FetchError, LedgerUnavailable, CorpusError, LedgerRefused, SafeReadError, MigrationRefused


(_LedgerCorrupt, _PackageError, _SuiteError, _FetchError, _LedgerUnavailable, _CorpusError, _LedgerRefused, _SafeReadError,
 _MigrationRefused) = _types()
from orchard.audio_package import AudioPackageError as _AudioPackageError  # noqa: E402
from orchard.model_config import ModelConfigError as _ModelConfigError  # noqa: E402
ALL_ITEMS = frozenset(i.name for i in ITEMS)         # a loader that lets every OSError propagate by design

def _preflight_config_json(base: Path):
    from orchard import preflight
    return preflight._config_json(base / "input.json")


def _preflight_index_problem(base: Path):
    from orchard import preflight
    return preflight._index_problem(base / "input.json")


def _audio_decision_load_category_rules(base: Path):
    from orchard import audio_decision
    return audio_decision.load_category_rules(base / "input.json")


READERS: list[Reader] = [
    Reader("preflight._config_json", _preflight_config_json),
    Reader("preflight._index_problem", _preflight_index_problem),
    Reader("audio_decision.load_category_rules", _audio_decision_load_category_rules, documented=(_SafeReadError,)),
    Reader("ledger.read_entries",_ledger_read_entries, suffix=b"\n", documented=(_LedgerCorrupt,)),
    Reader("telegraph_store.read_snapshot", _telegraph_store_read_snapshot, rel="project.db",
           documented=(_LedgerUnavailable, _LedgerRefused)),
    Reader("telegraph_migrate.read_archive_lines", _telegraph_migrate_read_archive_lines, suffix=b"\n", rel="telegraph.jsonl.archived",
           documented=(_SafeReadError,)),
    Reader("telegraph_migrate.migrate (dry run)", _telegraph_migrate_dry_run, suffix=b"\n", rel="telegraph.jsonl",
           documented=(_MigrationRefused,)),
    Reader("status.collect (hostile ledger)", _collect, suffix=b"\n", rel="ledger.jsonl", documented=(_LedgerCorrupt,)),
    Reader("status.collect (hostile control file)", _collect, rel="control", setup=_valid_ledger),
    Reader("status.collect (hostile pid file)", _collect, rel="supervisor.pid", setup=_valid_ledger),
    Reader("stages._load", _stages_load),
    Reader("stages._evidence_json", _stages_evidence_json, rel="evidence/input.json"),
    Reader("stages.hw_record_problem", _stages_hw_record, rel="tests/2/test-result.json"),
    Reader("stages._suite_reasons_in", _stages_suite_reasons_in),
    Reader("stages._read_json", _stages_read_json),
    Reader("stages.gate_delta", _stages_gate_delta, rel="stage/delta.json"),
    Reader("package.read_reported", _package_read_reported, rel="reported-results.json", documented=(_PackageError,)),
    Reader("package._json", _package_json, documented=(_PackageError,)),
    Reader("audio_package.read_json_file", _audio_package_read_json_file, documented=(_AudioPackageError,)),
    Reader("audio_package.read_facts (hostile result)", _audio_package_read_facts_result, rel="stages/2/audio_result.json",
           documented=(_AudioPackageError,)),
    Reader("audio_package.read_facts (hostile benchmark file)", _audio_package_read_facts_bench, rel="input.json", documented=(_AudioPackageError,)),
    Reader("audio_package.read_facts (hostile ledger)", _audio_package_read_facts_ledger, rel="ledger.jsonl", suffix=b"\n",
           documented=(_AudioPackageError,)),
    Reader("package.stage_2_vocab", _package_stage_2_vocab, rel="stages/2/model-dir-sources.json"),
    Reader("package._stage_2_graft_record", _package_graft_record, rel="stages/2/model-dir-sources.json"),
    Reader("bench_rerun.read_samples", _bench_rerun_read_samples, suffix=b"\n", rel="phase/results/run/samples_task.jsonl", documented=(ValueError,)),
    Reader("opinventory.load_table", _opinventory_load_table, documented=(ValueError,), os_error_ok=ALL_ITEMS),
    Reader("opinventory.load_ignored", _opinventory_load_ignored, documented=(ValueError,), os_error_ok=ALL_ITEMS),
    Reader("opinventory.load_dump", _opinventory_load_dump, rel="ttnn_ops-2.0.0.json"),
    Reader("opinventory.load_verified", _opinventory_load_verified),
    Reader("opvariants.load_data", _opvariants_load_data, documented=(ValueError,), os_error_ok=ALL_ITEMS),
    Reader("verdict._read_json", _verdict_read_json),
    Reader("sampling_choice._read_json", _sampling_choice_read_json),
    Reader("kept_coder.read", _kept_coder_read, rel="coder/kept.json"),
    Reader("speedwatch.Baselines._read", _speedwatch_baselines),
    Reader("run_view._read", _run_view_read),
    Reader("prompt_suite.load_suite", _prompt_suite_load, documented=(_SuiteError,)),
    Reader("supervisor._container_config", _supervisor_container_config, rel="swap_config.json"),
    Reader("supervisor._swap_cache", _supervisor_swap_cache, rel="swap_config.json"),
    Reader("card_use.tool_call_evidence", _card_use_tool_call_evidence, rel="stages/7/benchmark/speed.json"),
    Reader("transcripts.load_signature", _transcripts_load_signature, documented=(ValueError, TypeError, KeyError), os_error_ok=ALL_ITEMS),
    Reader("kernelprobe._read_agent_check", _kernelprobe_read_agent_check),
    Reader("skillsfresh.Cache", _skillsfresh_cache),
    Reader("toolcall_bench._rows", _toolcall_bench_rows, suffix=b"\n", documented=(ValueError,), os_error_ok=ALL_ITEMS),
    Reader("stock_runtime._load_config", _stock_runtime_load_config),
    Reader("model_config.read_model_config", _model_config_read, documented=(_ModelConfigError,)),
    Reader("model_config.dims (read then normalise)", _model_config_dims, documented=(_ModelConfigError,)),
    Reader("swap_draft.installed_containers", _swap_draft_installed_containers, rel=".cache/tt-model/installed.json"),
    Reader("nearest.read_containers", _nearest_read_containers, rel=".cache/tt-model/installed.json"),
    Reader("classify_bench.load_corpus", _classify_bench_load_corpus, documented=(_CorpusError,)),
    Reader("bench_session.load_items", _bench_session_load_items, documented=(ValueError,)),
    Reader("research.headers.read_model (hostile header)", _headers_read_model_header, feed="bytes"),
    Reader("research.headers.read_model (hostile index)", _headers_read_model_index, feed="bytes"),
    Reader("research.headers.shard_header", _headers_shard_header, feed="bytes", documented=(_FetchError, ValueError)),
    Reader("research.http.Fetcher.json", _http_fetcher_json, feed="bytes", documented=(_FetchError,)),
]

BY_NAME = {r.name: r for r in READERS}
assert len(BY_NAME) == len(READERS), "reader names must be unique"


# ------------------------------------------------------------------------------------------ readers that are wrong on purpose

def _selftest_safeio(base: Path):
    from orchard import safeio
    return safeio.read_json(base / "input.json", max_bytes=1 << 20)


def _selftest_passes_nan_on(base: Path):
    import json
    try:
        return json.loads((base / "input.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _selftest_raises_keyerror(base: Path):
    import json
    return json.loads((base / "input.json").read_text(encoding="utf-8"))["count"]


def _selftest_hangs_on_a_fifo(base: Path):
    with open(base / "input.json", "rb") as f:
        return len(f.read()) > 0


def _selftest_reads_without_a_cap(base: Path):
    return len((base / "input.json").read_bytes()) > 0


def _selftest_recurses(base: Path):
    import json
    try:
        return json.loads((base / "input.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


from orchard.safeio import SafeReadError as _SafeReadError  # noqa: E402

def _selftest_crashes(base: Path):
    import os
    os._exit(7)                                               # the child dies without a word


# These prove that the instrument can report a failure (and a pass). They are not in READERS.
SELFTEST = {r.name: r for r in [
    Reader("selftest.safeio", _selftest_safeio, documented=(_SafeReadError,)),
    Reader("selftest.passes_nan_on", _selftest_passes_nan_on),
    Reader("selftest.raises_keyerror", _selftest_raises_keyerror),
    Reader("selftest.hangs_on_a_fifo", _selftest_hangs_on_a_fifo),
    Reader("selftest.reads_without_a_cap", _selftest_reads_without_a_cap),
    Reader("selftest.recurses", _selftest_recurses),
    Reader("selftest.crashes", _selftest_crashes),
]}

# The defects this registry found, by the GitHub issue that tracks each one. A (reader, item) pair listed here is marked
# xfail(strict=True) by tests/test_hostile_readers.py, so a reader that is fixed turns its test red until the entry is removed.
KNOWN_DEFECTS: dict[int, dict[str, list[str]]] = {
    135: {          # NaN, Infinity, 1e999 and lone surrogates are accepted and returned
        'stages._evidence_json': ['nan_in_object', 'lone_surrogate_key'],
        'stages._load': ['nan_in_object', 'lone_surrogate_key'],
        'stages._read_json': ['huge_exponent', 'nan', 'infinity', 'negative_infinity', 'nan_in_object', 'lone_surrogate', 'lone_surrogate_key'],
    },
    136: {          # very deep nesting or a long integer raises an undocumented exception, or a deep structure is returned unbounded
        'stages._evidence_json': ['nest_1000_object', 'nest_100000'],
        'stages._load': ['nest_1000_object', 'nest_100000'],
        'stages._read_json': ['nest_1000', 'nest_1000_object', 'nest_100000'],
        'stages._suite_reasons_in': ['nest_100000'],
        'stages.gate_delta': ['nest_100000'],
        'stages.hw_record_problem': ['nest_100000'],
    },
    137: {          # the file is read with no size limit (a 50 MB file is held in memory; a link to /dev/zero is read until memory runs out)
        'stages._load': ['big_string_50mb', 'link_to_dev_zero'],
        'stages._read_json': ['big_string_50mb', 'link_to_dev_zero'],
        'stages._suite_reasons_in': ['big_string_50mb', 'link_to_dev_zero'],
        'stages.gate_delta': ['big_string_50mb', 'link_to_dev_zero'],
        'stages.hw_record_problem': ['big_string_50mb', 'link_to_dev_zero'],
    },
    138: {          # the reader blocks forever on a FIFO
        'stages._evidence_json': ['fifo'],
        'stages._load': ['fifo'],
        'stages._read_json': ['fifo'],
        'stages._suite_reasons_in': ['fifo'],
        'stages.gate_delta': ['fifo'],
        'stages.hw_record_problem': ['fifo'],
    },
    139: {          # an undocumented exception type for a directory, a symlink loop or invalid UTF-8
        'stages._evidence_json': ['link_loop'],
    },
    140: {          # valid JSON that is not an object raises AttributeError
    },
}

KNOWN_DEFECT_TEXT = {
    135: 'NaN, Infinity, 1e999 and lone surrogates are accepted and returned',
    136: 'very deep nesting or a long integer raises an undocumented exception, or a deep structure is returned unbounded',
    137: 'the file is read with no size limit (a 50 MB file is held in memory; a link to /dev/zero is read until memory runs out)',
    138: 'the reader blocks forever on a FIFO',
    139: 'an undocumented exception type for a directory, a symlink loop or invalid UTF-8',
    140: 'valid JSON that is not an object raises AttributeError',
}

XFAIL_BY_PAIR = {(name, item): number for number, readers in KNOWN_DEFECTS.items() for name, items in readers.items() for item in items}
assert all(name in BY_NAME for readers in KNOWN_DEFECTS.values() for name in readers), "a known defect names a reader that is not registered"

# Modules that parse JSON and have no reader in READERS yet: the backlog. tests/test_hostile_readers.py keeps this list exact.
UNREGISTERED_MODULES = (
    '_proofs/telegraph_sqlite.py',
    'adapters/gozer.py',
    'agent.py',
    'audio_bars.py',
    'audio_cli.py',
    'audio_facts.py',
    'audio_reference.py',
    'audio_reference_env.py',
    'audio_remote.py',
    'audio_runtime.py',
    'bench_control.py',
    'bench_proxy.py',
    'boards_seen.py',
    'caches.py',
    'canary.py',
    'cli.py',
    'demo_env.py',
    'fake_server.py',
    'graft_trial.py',
    'granite_package.py',
    'handoff.py',
    'hardware_check.py',
    'hwtests.py',
    'hwtool.py',
    'kernelladder.py',
    'lab.py',
    'labclient.py',
    'measure_serving.py',
    'narrate.py',
    'operator_checks.py',
    'opverify.py',
    'package_card.py',
    'package_cmd.py',
    'qwen35_package.py',
    'reported_cmd.py',
    'research/ports.py',
    'rolefit.py',
    'safeio.py',
    'sizing.py',
    'suite_control.py',
    'telegraph.py',
    'telegraph_cmd.py',
    'text_cli.py',
    'text_package.py',
    'text_remote.py',
    'text_runtime.py',
    'text_stages.py',
    'text_stages.py',
    'ttnn_dump.py',
    'verify_package.py',
    'virtualspend.py',
    'webui.py',
)
