# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 5 of a full port, the supervisor's prepare step (issue 329). What `orchard/full_port_serve.py` describes, as code that writes files.

`prepare` puts the stage directory in the state the one hardware command needs: the port that passed stage 2 or 3 copied to `stages/5/port/`, the
shipped templates `port_serve.py` and `port_check.py`, the acceptance corpus, the config and `hw_test.json`. It returns what it wrote with the
sha256 of each file, so the supervisor can put them in the ledger, and the reasons the stage cannot run when something is missing. It starts no
process and talks to no chip.

The port copy takes regular files only. A link, a device or a FIFO, more than `MAX_FILES` files, a file over `MAX_FILE_BYTES`, more than
`MAX_ENTRIES` entries of any kind or more than `MAX_DEPTH` directory levels stop the copy and name the reason. The files are read with the bounded
reader, written through `safeio.atomic_write` and hashed from the bytes that were read, which are the bytes written. The gate compares those hashes
with the accepted stage's record, so a copy that changed a file in transit is refused later.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from dataclasses import dataclass, field
from pathlib import Path

from orchard import full_port, full_port_serve, safeio

MAX_FILES = 200
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_ENTRIES = 1000
MAX_DEPTH = 8
JSON_MAX = 16 * 1024 * 1024


@dataclass
class Prepared:
    problems: list[str] = field(default_factory=list)     # why the stage cannot run; empty when it can
    source: int | None = None
    corpus_sha256: str | None = None
    positions: int = 0
    port_files: list[dict] = field(default_factory=list)  # {"path", "sha256"} of each copied file
    pinned: dict = field(default_factory=dict)            # {"template", "worker", "config", "corpus"}: the sha256 of each file written


def _list_port(src) -> list[tuple[str, ...]]:
    """The regular files under the pinned source directory `src`, as sorted name tuples, or ValueError naming why the directory is refused.

    The walk is `safeio.scan_beneath`: it streams each directory one entry at a time (a directory with a million names is never held whole),
    spends one budget of MAX_ENTRIES on every entry of every directory, opens each subdirectory beneath its parent with no link followed, and
    refuses at MAX_DEPTH directory levels. `__pycache__` and `.pyc` files are skipped."""
    found: list[tuple[str, ...]] = []

    def visit(entry) -> None:
        parts = entry.parts
        if "__pycache__" in parts or parts[-1].endswith(".pyc"):
            return
        rel = "/".join(parts)
        if entry.kind == "link":
            raise ValueError(f"{rel} is a link")
        if entry.kind == "dir":
            return
        if entry.kind != "file":
            raise ValueError(f"{rel} is not a regular file")
        if len(found) >= MAX_FILES:
            raise ValueError(f"more than {MAX_FILES} files")
        found.append(parts)

    result = safeio.scan_beneath(src, visit, max_entries=MAX_ENTRIES, max_depth=MAX_DEPTH)
    if result.entries_cut:
        raise ValueError(f"more than {MAX_ENTRIES} entries")
    if result.depth_cut:
        raise ValueError(f"more than {MAX_DEPTH} directory levels")
    if result.unreadable:
        raise ValueError(f"a directory in it could not be read ({result.unreadable[0][1]})")
    return sorted(found)


def _clear_port(run, stage_parts: tuple[str, ...]) -> None:
    """Remove what stands at `port` in the stage directory `stage_parts` (beneath the pin `run`): a real directory with its tree, or a link or a
    file by itself. The stage directory is opened with no link followed at any name and the removal goes through its descriptor, so a link
    planted at `stages/5` is refused (SafeReadError) and a link planted at `port` is removed and never followed."""
    fd = safeio.open_beneath(run.fd, stage_parts)
    try:
        try:
            mode = os.lstat("port", dir_fd=fd).st_mode
        except FileNotFoundError:
            return
        if stat.S_ISDIR(mode):
            shutil.rmtree("port", dir_fd=fd)         # a port left by an earlier, unfinished attempt of this stage directory
        else:
            os.unlink("port", dir_fd=fd)
    finally:
        os.close(fd)


def copy_port(run_dir: Path, source: int, stage_dir: Path) -> tuple[list[dict], str | None]:
    """Copy `stages/<source>/port/` to `<stage_dir>/port/`, replacing a port already there. Returns ([{"path", "sha256"}], None) or ([], why).

    Both ends are anchored beneath one descriptor of the run directory (review of PR 490): the source is opened with no link followed at any
    name, so a link planted at `stages/2` is refused, and the destination is removed and written only through descriptors, so a link planted
    at `stages/5` cannot send the removal or a write outside the run."""
    run_dir, stage_dir = Path(run_dir), Path(stage_dir)
    src_parts = ("stages", str(source), "port")
    try:
        stage_parts = stage_dir.relative_to(run_dir).parts
    except ValueError:
        return [], "the stage directory is not inside the run directory"
    if not stage_parts:
        return [], "the stage directory is not inside the run directory"
    dst_parts = (*stage_parts, "port")
    not_dir = f"stages/{source}/port/ is not a directory"
    try:
        with safeio.pinned_root(run_dir) as run:
            try:
                src = run.child(src_parts)
            except safeio.SafeReadError:
                return [], not_dir
            with src:
                try:
                    found = _list_port(src)
                except ValueError as exc:
                    return [], f"stages/{source}/port/ cannot be copied: {exc}"[:300]
                if not found:
                    return [], f"stages/{source}/port/ holds no file"
                safeio.mkdir_beneath(run.fd, stage_parts)
                _clear_port(run, stage_parts)
                out = []
                for parts in found:
                    data = safeio.read_beneath(run.fd, (*src_parts, *parts), max_bytes=MAX_FILE_BYTES)
                    safeio.write_beneath(run.fd, (*dst_parts, *parts), data, mode=0o644)
                    out.append({"path": f"stages/{full_port_serve.STAGE}/port/{'/'.join(parts)}", "sha256": hashlib.sha256(data).hexdigest()})
                return out, None
    except (OSError, safeio.SafeReadError) as exc:
        return [], f"stages/{source}/port/ cannot be copied: {getattr(exc, 'reason', None) or exc.__class__.__name__}"[:300]


def _put(path: Path, data: bytes, mode: int = 0o644) -> str:
    safeio.atomic_write(path, data, mode=mode)
    return hashlib.sha256(data).hexdigest()


def prepare(run_dir, stage_dir, entries: list[dict], *, snapshot, orchard_dir) -> Prepared:
    """Write the stage 5 directory. `entries` are the ledger's entries (they say which stage passed and which hash its record had), `snapshot`
    the run's model directory, `orchard_dir` the checkout whose shipped templates are copied."""
    run_dir, stage_dir = Path(run_dir), Path(stage_dir)
    out = Prepared()
    if not snapshot:
        out.problems.append("the run has no model snapshot (inputs `model=`), so the serving check has nothing to load")
        return out
    out.source = full_port_serve.source_stage(entries)
    if out.source is None:
        out.problems.append("neither stage 3 nor stage 2 of this run ended with a pass, so there is no port to serve")
        return out
    accepted = full_port_serve.accepted_sha(entries, out.source, f"stages/{out.source}/{full_port.RECORD_FILE}")
    if accepted is None:
        out.problems.append(f"the ledger holds no sha256 for stage {out.source}'s port-check record among the evidence its gate accepted")
        return out
    corpus, problems = full_port_serve.build_corpus(run_dir, out.source, accepted=accepted)
    if corpus is None:
        out.problems.extend(problems)
        return out
    templates = Path(orchard_dir) / "orchard" / "skills" / "full-port-templates"
    try:
        stage_dir.mkdir(parents=True, exist_ok=True)
        files, why = copy_port(run_dir, out.source, stage_dir)
        if why:
            out.problems.append(why)
            return out
        out.port_files = files
        for name in (full_port_serve.TEMPLATE_NAME, full_port_serve.WORKER_NAME):
            data = safeio.read_bytes(templates / name, max_bytes=JSON_MAX)
            out.pinned["template" if name == full_port_serve.TEMPLATE_NAME else "worker"] = _put(stage_dir / name, data, mode=0o755)
        data = full_port_serve.corpus_bytes(corpus)
        out.corpus_sha256 = out.pinned["corpus"] = _put(stage_dir / full_port_serve.CORPUS_NAME, data)
        out.positions = full_port_serve.positions(corpus)
        config = full_port_serve.config_for(str(Path(snapshot).absolute()), out.source, out.corpus_sha256)
        if not config["passkeys"]:
            out.problems.append(f"the model's context ({config['max_len']} tokens) is too short for two passkey lengths")
            return out
        out.pinned["config"] = _put(stage_dir / full_port_serve.CONFIG_NAME, (json.dumps(config, indent=2) + "\n").encode())
        _put(stage_dir / "hw_test.json", (json.dumps({"command": full_port_serve.command(), "deadline_s": full_port_serve.DEADLINE_S}) + "\n").encode())
        # The supervisor's hardware phase reads a handoff note beside the test (orchard/handoff.py NOTE_KEYS). No agent writes one in this stage.
        note = {"goal": "show the port that passed stage %d behind the chat server: replay the acceptance corpus, passkey, canary" % out.source,
                "stage": full_port_serve.STAGE, "evidence": ["stages/5/evidence/port-serve.json", "stages/5/corpus.json"],
                "next_action": "read the test output and evidence/port-serve.json", "check_on_return": "the test exit code and the READY line"}
        _put(stage_dir / "handoff.json", (json.dumps(note) + "\n").encode())
    except (OSError, safeio.SafeReadError) as exc:
        out.problems.append(f"the stage directory could not be written ({exc.__class__.__name__})")
    return out

