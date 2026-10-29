# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A virtual spend for local models: what the local coder's token use would cost at a hosted model's price.

The number is for comparison with the real spend on the outside helper (orchard/telegraph.py). It is virtual:

  * the agent logs hold the prompt tokens of the whole conversation on every turn, so the prompt count is large. A
    hosted provider with prompt caching bills most of those repeated tokens for much less, so the figure is an upper
    bound at a cache-less price. Use it as a comparison figure only;
  * it is never added to the real spend, and nothing here writes a file, opens a chip or sends a request.

Sources that count as local (each is a file the local harness wrote):
  * `log/*.jsonl` agent logs (a kernel-probe attempt, or a bring-up stage): one object per turn with `usage`;
  * `requests.jsonl` of the benchmark proxy: one object per request with `prompt_tokens`, `completion_tokens`, `latency_s`.
A turn that came from a hosted helper (a `call_id`, or `origin`, `source` or `helper` naming one) is skipped and counted.
`telegraph.jsonl` is the helper's own ledger and is never read here.

Reading is bounded and never raises: a malformed line, bytes that are not UTF-8, a line over the size cap, a turn with no
usage and a file that cannot be read are counted in `skipped`, and the summary says so.
"""
from __future__ import annotations

import difflib
import errno
import json
import math
import os
import re
import stat
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

KEYS = {"reference", "price_in", "price_out", "extra"}
EXTRA_KEYS = {"name", "price_in", "price_out"}
DEFAULT_REFERENCE = "claude-haiku-5-5"
DEFAULT_PRICE_IN = 0.10
DEFAULT_PRICE_OUT = 0.50
NAME_MAX = 60
MAX_EXTRA = 8
LABEL = "This is a virtual figure: local tokens priced at a hosted model's list rate, never added to the real helper spend"

MAX_FILES = 5000                     # log files read in one scan
MAX_DEPTH = 8
MAX_WALK_ENTRIES = 200_000        # directory entries taken in one scan of one folder tree, across every folder in it
MAX_FILE_BYTES = 128 * 1024 * 1024   # a larger file is read up to this many bytes and the rest is counted as not read
MAX_TOTAL_BYTES = 1024 * 1024 * 1024
MAX_LINE_BYTES = 2 * 1024 * 1024     # a longer line is not parsed; its usage is looked for in its last bytes
TAIL_BYTES = 8192
HEAD_BYTES = 4096                    # the start of an over-long line, searched for helper markers and the agent's turn shape
MAX_TOKENS_PER_TURN = 10 ** 9        # a bigger number is a corrupt record

HELPER_NAMES = {"telegraph", "helper", "hosted", "outside"}
# The helper markers `_is_helper` reads, as byte patterns for the first and last bytes of an over-long line.
_HELPER_BYTES = re.compile(
    rb'"call_id"\s*:|"(?:origin|source)"\s*:\s*"(?:' + b"|".join(n.encode() for n in sorted(HELPER_NAMES)) + rb')"'
    rb'|"helper"\s*:\s*(?:true|[1-9"\[{])', re.IGNORECASE)
AGENT_TURN_START = b'{"turn"'


class ConfigError(ValueError):
    """The [virtual_spend] table is wrong; the message names the key."""


@dataclass(frozen=True)
class Config:
    reference: str = DEFAULT_REFERENCE
    price_in: float = DEFAULT_PRICE_IN
    price_out: float = DEFAULT_PRICE_OUT
    extra: tuple = ()                # ((name, price_in, price_out), ...)

    def references(self) -> list[tuple[str, float, float]]:
        return [(self.reference, self.price_in, self.price_out), *self.extra]


def _price(value, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ConfigError(f"virtual_spend.{key} must be a number of dollars per million tokens, zero or more")
    return float(value)


def _name(value, key: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > NAME_MAX:
        raise ConfigError(f"virtual_spend.{key} must be a name of 1 to {NAME_MAX} characters")
    return value.strip()


def load(table) -> Config:
    """The `[virtual_spend]` table as a Config. No table gives the defaults."""
    if table is None:
        return Config()
    if not isinstance(table, dict):
        raise ConfigError("virtual_spend must be a table")
    for key in table:
        if key not in KEYS:
            near = difflib.get_close_matches(str(key), sorted(KEYS), n=1)
            raise ConfigError(f"virtual_spend: unknown key {key!r}" + (f"; did you mean {near[0]!r}?" if near else ""))
    reference = _name(table["reference"], "reference") if "reference" in table else DEFAULT_REFERENCE
    price_in = _price(table["price_in"], "price_in") if "price_in" in table else DEFAULT_PRICE_IN
    price_out = _price(table["price_out"], "price_out") if "price_out" in table else DEFAULT_PRICE_OUT
    extra_raw = table.get("extra", [])
    if not isinstance(extra_raw, list):
        raise ConfigError("virtual_spend.extra must be a list of tables with name, price_in and price_out")
    if len(extra_raw) > MAX_EXTRA:
        raise ConfigError(f"virtual_spend.extra has more than {MAX_EXTRA} entries")
    extra, seen = [], {reference}
    for item in extra_raw:
        if not isinstance(item, dict):
            raise ConfigError("virtual_spend.extra must be a list of tables with name, price_in and price_out")
        for key in item:
            if key not in EXTRA_KEYS:
                raise ConfigError(f"virtual_spend.extra: unknown key {key!r}")
        for key in EXTRA_KEYS:
            if key not in item:
                raise ConfigError(f"virtual_spend.extra needs {key}")
        name = _name(item["name"], "extra.name")
        if name in seen:
            raise ConfigError(f"virtual_spend names {name!r} twice")
        seen.add(name)
        extra.append((name, _price(item["price_in"], "extra.price_in"), _price(item["price_out"], "extra.price_out")))
    return Config(reference=reference, price_in=price_in, price_out=price_out, extra=tuple(extra))


def load_file(path) -> Config:
    """The `[virtual_spend]` table of a TOML file; a file with no table gives the defaults."""
    path = Path(path)
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"cannot read {str(path)!r}: {exc}") from exc
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise ConfigError(f"cannot parse {str(path)!r}: {exc}") from exc
    return load(raw.get("virtual_spend"))


def cost(prompt_tokens: int, completion_tokens: int, price_in: float, price_out: float) -> float:
    """Dollars for the tokens at a price in dollars per million tokens."""
    return (prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000


# ------------------------------------------------------------------------------------------------------ reading

def _skipped() -> dict:
    return {"malformed_lines": 0, "no_usage": 0, "helper_turns": 0, "long_lines": 0, "long_unverified": 0, "not_utf8_lines": 0,
            "unreadable_files": 0, "oversize_files": 0, "files_not_read": 0, "walk_cut": 0, "depth_cut": 0, "links": 0}


def _count(value) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > MAX_TOKENS_PER_TURN:
        return None
    return value


def _is_helper(rec: dict) -> bool:
    if "call_id" in rec:
        return True
    for key in ("origin", "source"):
        if isinstance(rec.get(key), str) and rec[key].lower() in HELPER_NAMES:
            return True
    return bool(rec.get("helper"))


def _long_is_helper(head: bytes, tail: bytes) -> bool:
    return bool(_HELPER_BYTES.search(head) or _HELPER_BYTES.search(tail))


def _agent_turn(rec: dict):
    """(prompt, completion) of an agent-log turn, or None when it has no usable usage."""
    usage = rec.get("usage")
    if not isinstance(usage, dict):
        return None
    p, c = _count(usage.get("prompt_tokens")), _count(usage.get("completion_tokens"))
    if p is None or c is None:
        return None
    return p, c


def _seconds(value) -> float | None:
    """A non-negative finite number of seconds, or None. A JSON integer too large for a float is no time."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        seconds = float(value)
    except OverflowError:
        return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def _request(rec: dict):
    """(prompt, completion, latency) of a benchmark proxy record."""
    p, c = _count(rec.get("prompt_tokens")), _count(rec.get("completion_tokens"))
    if p is None or c is None:
        return None
    return p, c, _seconds(rec.get("latency_s")) or 0.0


def _room(read: int, budget: list, want: int) -> int:
    """How many bytes may still be read: the wanted size, cut to what is left of the file and the scan budgets."""
    return min(want, MAX_FILE_BYTES - read, budget[0])


def _more(fh) -> bool:
    """True when the file has more bytes after the current position (the position does not move)."""
    try:
        return bool(fh.peek(1))
    except (OSError, ValueError, AttributeError):
        return False


MAX_ATTEMPT_BYTES = 1024 * 1024      # an attempt.json larger than this is not read


_FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
_LINK_ERRNOS = (errno.ELOOP, errno.EMLINK, errno.ENOTDIR)


class _Pin:
    """The scan root, opened once without following a link and held as a descriptor for the whole scan. Every file is then opened
    relative to this descriptor, one directory component at a time, each with O_DIRECTORY | O_NOFOLLOW. A directory swapped for a
    link after discovery is refused at its component, and a root swapped after the scan began is not followed."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.fd = os.open(self.root, _DIR_FLAGS)

    def close(self):
        try:
            os.close(self.fd)
        except OSError:
            pass

    def child(self, rel: tuple) -> "_Pin":
        """A pin on the folder `rel` (path components) beneath this one. Each component is opened with O_DIRECTORY | O_NOFOLLOW
        relative to the descriptor before it, so a link anywhere on the way is refused and no ancestor is followed. Raises OSError."""
        fd, owned = self.fd, False
        try:
            for part in rel:
                nxt = os.open(part, _DIR_FLAGS, dir_fd=fd)
                if owned:
                    os.close(fd)
                fd, owned = nxt, True
            if not owned:
                fd = os.dup(self.fd)
        except OSError:
            if owned:
                os.close(fd)
            raise
        pin = _Pin.__new__(_Pin)
        pin.root, pin.fd = self.root.joinpath(*rel), fd
        return pin

    def open(self, path: Path, flags: int) -> int:
        """A descriptor for `path` (which must lie under the root), opened beneath the pinned root. Raises OSError."""
        try:
            parts = Path(path).relative_to(self.root).parts
        except ValueError as exc:
            raise OSError(errno.ELOOP, "outside the scan root") from exc
        if not parts:
            raise OSError(errno.EISDIR, "the scan root is not a file")
        fd, owned = self.fd, False
        try:
            for part in parts[:-1]:
                nxt = os.open(part, _DIR_FLAGS, dir_fd=fd)
                if owned:
                    os.close(fd)
                fd, owned = nxt, True
            return os.open(parts[-1], flags, dir_fd=fd)
        finally:
            if owned:
                os.close(fd)


def open_pin(root) -> "_Pin | None":
    """Pin a folder (a runs folder) without following a link, for `scan(..., beneath=pin)`. None when it cannot be opened. The caller
    closes it. A scan of each folder beneath one pin follows no link on the way from the pinned folder down."""
    try:
        return _Pin(Path(root))
    except OSError:
        return None


def _open_log(path: Path, skipped: dict, pin: "_Pin | None" = None):
    """Open a log for reading beneath the pinned scan root without following a link, and keep it only if the opened descriptor is a
    regular file. The open is the check: a file or a directory swapped for a link after discovery is refused here. Returns a binary
    file object, or None after counting why (a link, a swapped directory or a file that is not regular is a skipped link; anything
    else is an unreadable file). With no pin, the log's own folder is pinned for this one open."""
    own = pin is None
    try:
        if own:
            pin = _Pin(Path(path).parent)
        try:
            fd = pin.open(path, _FILE_FLAGS)
        except OSError as exc:
            skipped["links" if exc.errno in _LINK_ERRNOS else "unreadable_files"] += 1
            return None
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                os.close(fd)
                skipped["links"] += 1
                return None
            return os.fdopen(fd, "rb")
        except OSError:
            try:
                os.close(fd)
            except OSError:
                pass
            skipped["unreadable_files"] += 1
            return None
    except OSError:
        skipped["unreadable_files"] += 1
        return None
    finally:
        if own and pin is not None:
            pin.close()


def _read_capped(path: Path, cap: int, pin: "_Pin") -> bytes | None:
    """The bytes of a small regular file, opened beneath the pinned root without following a link and read with at most cap+1 bytes
    asked for. None when the file is a link, is not regular, cannot be read or holds more than `cap` bytes."""
    try:
        fd = pin.open(path, _FILE_FLAGS)
    except OSError:
        return None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return None
        data = b""
        while len(data) <= cap:
            chunk = os.read(fd, min(65536, cap + 1 - len(data)))
            if not chunk:
                break
            data += chunk
        return data if len(data) <= cap else None
    except OSError:
        return None
    finally:
        os.close(fd)


def _lines(path: Path, skipped: dict, budget: list, pin: "_Pin | None" = None):
    """Yield ("rec", dict) for each line that parses, and ("tail", bytes) for a line over the cap. Every read is limited to
    what is left of the file budget and the scan budget. A record that a budget cuts off is dropped and counted in
    `oversize_files`, so its tail is never priced."""
    read = 0
    fh = _open_log(path, skipped, pin)
    if fh is None:
        return
    with fh:
        while True:
            want = MAX_LINE_BYTES + 1
            room = _room(read, budget, want)
            if room <= 0:
                if _more(fh):
                    skipped["oversize_files"] += 1
                return
            try:
                line = fh.readline(room)
            except OSError:
                skipped["unreadable_files"] += 1
                return
            if not line:
                return
            read += len(line)
            budget[0] -= len(line)
            complete = line.endswith(b"\n")
            if not complete and len(line) == room and room < want and _more(fh):
                skipped["oversize_files"] += 1                   # a budget ended in the middle of this line
                return
            if len(line) > MAX_LINE_BYTES and not complete:
                head, tail = line[:HEAD_BYTES], line[-TAIL_BYTES:]
                ended = False
                while True:                                      # drain the rest of the line, keeping only its end
                    room = _room(read, budget, MAX_LINE_BYTES)
                    if room <= 0:
                        ended = not _more(fh)                    # a line that ends with the file is whole
                        break
                    try:
                        more = fh.readline(room)
                    except OSError:
                        break
                    if not more:
                        ended = True
                        break
                    read += len(more)
                    budget[0] -= len(more)
                    tail = (tail + more)[-TAIL_BYTES:]
                    if more.endswith(b"\n"):
                        ended = True
                        break
                if not ended:
                    skipped["oversize_files"] += 1
                    return
                yield "tail", (head, tail)
                continue
            if not line.strip():
                continue
            try:
                text = line.decode("utf-8")
            except UnicodeDecodeError:
                skipped["not_utf8_lines"] += 1
                text = line.decode("utf-8", errors="replace")      # a bad byte in a note does not hide the turn's counts
            try:
                rec = json.loads(text)
            except (ValueError, RecursionError):
                skipped["malformed_lines"] += 1
                continue
            if not isinstance(rec, dict):
                skipped["malformed_lines"] += 1
                continue
            yield "rec", rec


def _tail_usage(tail: bytes):
    """(prompt, completion) from the last `"usage"` key in the end of an over-long line, or None. The agent writes `usage` as the
    last key of a turn, so after the usage object only the closing brace of the record may follow. The object is parsed with a
    JSON decoder, so nested details (`completion_tokens_details`) are read; a suffix that was cut off is not."""
    at = tail.rfind(b'"usage"')
    if at < 0:
        return None
    rest = tail[at + len(b'"usage"'):].decode("utf-8", errors="replace").lstrip()
    if not rest.startswith(":"):
        return None
    rest = rest[1:].lstrip()
    try:
        usage, end = json.JSONDecoder().raw_decode(rest)
    except (ValueError, RecursionError):
        return None
    if rest[end:].strip() != "}":
        return None
    return _agent_turn({"usage": usage})


def _list_dir(source, budget: list) -> tuple[list, bool]:
    """([(name, is_link, is_dir, is_regular, link_to_dir)], cut): the entries of one folder (a path, or a directory descriptor), taken while
    streaming os.scandir. Every entry taken is counted against the shared budget; when it is spent the listing stops and `cut` is
    True. A folder that cannot be listed, or an entry that cannot be examined, gives what was read so far and appends one
    "unreadable" item to `budget` (the caller counts and removes them; see `_walk`), so the scan is incomplete and says so (issue 149)."""
    out, cut = [], False
    try:
        with os.scandir(source) as listing:
            for entry in listing:
                if budget[0] <= 0:
                    cut = True
                    break
                budget[0] -= 1
                try:
                    link = entry.is_symlink()
                    out.append((entry.name, link, (not link) and entry.is_dir(follow_symlinks=False),
                                (not link) and entry.is_file(follow_symlinks=False),
                                link and entry.is_dir()))               # a link that points at a folder, decided here, not by path later
                except OSError:
                    budget.append("unreadable")
                    continue
    except OSError:
        budget.append("unreadable")
    return sorted(out), cut


def _find_files(root: Path, skipped: dict, pin: "_Pin | None" = None):
    """(path, kind) for each local log file under root, kind 'agent' or 'bench'. Links are not followed. Folders are listed through
    descriptors opened beneath the pinned root (`pin`, or a pin made here), one component at a time, so the listing and the later
    reads see the same tree whatever happens to the path names. The tree is walked as a stream: one entry budget (MAX_WALK_ENTRIES) is
    shared by every folder, a folder is sorted only after its own entries were taken, and the walk stops opening folders once more
    than MAX_FILES log files were seen. A spent budget is counted in `walk_cut`."""
    own = pin is None
    if own:
        try:
            pin = _Pin(Path(root))
        except OSError:
            return []
    try:
        return _walk(pin, skipped)
    finally:
        if own:
            pin.close()


def _walk(pin: "_Pin", skipped: dict):
    found, seen = [], 0
    budget = [MAX_WALK_ENTRIES]
    stack = [((), 0)]
    while stack:
        rel, depth = stack.pop()
        if budget[0] <= 0:                                           # no folder is opened once the entry budget is spent
            skipped["walk_cut"] += 1                                 # the rest of the traversal is abandoned, so the scan is incomplete
            break
        try:
            here = pin.child(rel)
        except OSError as exc:                                       # a folder swapped for a link, or gone, since it was listed
            skipped["links" if exc.errno in _LINK_ERRNOS else "unreadable_files"] += 1
            continue
        try:
            entries, cut = _list_dir(here.fd, budget)
        finally:
            here.close()
        while len(budget) > 1:                                       # failures that _list_dir reported: each makes the scan incomplete
            budget.pop()
            skipped["unreadable_files"] += 1
        if cut:
            skipped["walk_cut"] += 1
        in_log = (rel[-1] if rel else pin.root.name) == "log"
        subdirs = []
        for name, link, is_dir, is_regular, link_to_dir in entries:
            if link and not _is_log_name(name, in_log):
                if link_to_dir:
                    skipped["links"] += 1
                continue
            if is_dir:
                if depth < MAX_DEPTH:
                    subdirs.append(name)
                else:
                    skipped["depth_cut"] += 1                        # logs below the limit are not read: the scan is incomplete
                continue
            kind = _log_kind(name, in_log)
            if kind is None:
                continue
            if link or not is_regular:
                skipped["links"] += 1
                continue
            seen += 1
            if seen > MAX_FILES:
                skipped["files_not_read"] += 1
                continue
            found.append((pin.root.joinpath(*rel, name), kind))
        if seen > MAX_FILES:
            break                                                    # no folder is opened after the file cap
        for name in reversed(subdirs):                               # popped in name order
            stack.append((rel + (name,), depth + 1))
    return found


def _log_kind(name: str, in_log: bool):
    return "bench" if name == "requests.jsonl" else "agent" if in_log and name.endswith(".jsonl") else None


def _is_log_name(name: str, in_log: bool) -> bool:
    return _log_kind(name, in_log) is not None


def _attempt_wall(log_dir: Path, pin: "_Pin") -> float:
    """The agent's wall seconds a kernel-probe attempt recorded beside its log directory (`attempt.json`), else 0. The file is opened
    without following links and read with a size limit, so a swapped link or a large file gives no time."""
    raw = _read_capped(log_dir.parent / "attempt.json", MAX_ATTEMPT_BYTES, pin)
    if raw is None:
        return 0.0
    try:
        v = json.loads(raw.decode("utf-8")).get("agent_wall_s")
    except (ValueError, AttributeError, UnicodeDecodeError, RecursionError):
        return 0.0
    return _seconds(v) or 0.0


def scan(root, beneath: "_Pin | None" = None) -> dict:
    """Count the local turns under `root` (a directory). With `beneath` (a pin from `open_pin` on an ancestor of `root`), the folder is
    opened relative to that descriptor one component at a time, so no ancestor of `root` is followed. Raises nothing. `agent_wall_s` (from a kernel-probe attempt.json) and
    `request_s` (the benchmark proxy's request times, summed across concurrent requests, so not wall time) are None when unknown."""
    root = Path(root)
    skipped = _skipped()
    out = {"turns": 0, "prompt_tokens": 0, "completion_tokens": 0, "agent_wall_s": None, "request_s": None, "files": 0,
           "skipped": skipped, "sources": {"agent": 0, "benchmark": 0}, "complete": True}
    try:
        if beneath is not None:
            pin = beneath.child(root.relative_to(beneath.root).parts)
        else:
            if root.is_symlink() or not root.is_dir():
                out["complete"] = False
                return out
            pin = _Pin(root)                                        # opened once; every file is opened beneath this descriptor
    except (OSError, ValueError) as exc:
        out["complete"] = False
        if isinstance(exc, OSError) and exc.errno in _LINK_ERRNOS:
            skipped["links"] += 1
        return out
    try:
        _scan_tree(root, pin, out, skipped)
    finally:
        pin.close()
    return out


def _scan_tree(root: Path, pin: _Pin, out: dict, skipped: dict) -> None:
    budget = [MAX_TOTAL_BYTES]
    wall, wall_known, request_s, request_known, attempts = 0.0, False, 0.0, False, set()
    for path, kind in _find_files(root, skipped, pin):
        out["files"] += 1
        for what, payload in _lines(path, skipped, budget, pin):
            if what == "tail":
                skipped["long_lines"] += 1
                head, tail = payload
                if kind != "agent":
                    skipped["no_usage"] += 1
                    continue
                if _long_is_helper(head, tail):                  # the same filter a line under the cap gets, from its ends
                    skipped["helper_turns"] += 1
                    continue
                if not head.lstrip().startswith(AGENT_TURN_START):
                    skipped["long_unverified"] += 1               # its source cannot be established: not priced
                    continue
                got = _tail_usage(tail)
                if got is None:
                    skipped["no_usage"] += 1
                    continue
                out["turns"] += 1
                out["prompt_tokens"] += got[0]
                out["completion_tokens"] += got[1]
                out["sources"]["agent"] += 1
                continue
            if _is_helper(payload):
                skipped["helper_turns"] += 1
                continue
            if kind == "agent":
                got = _agent_turn(payload)
                lat = 0.0
            else:
                req = _request(payload)
                got = req[:2] if req else None
                lat = req[2] if req else 0.0
                if req:
                    request_known = True
            if got is None:
                skipped["no_usage"] += 1
                continue
            out["turns"] += 1
            out["prompt_tokens"] += got[0]
            out["completion_tokens"] += got[1]
            out["sources"]["agent" if kind == "agent" else "benchmark"] += 1
            request_s += lat
        if kind == "agent" and path.parent not in attempts:
            attempts.add(path.parent)
            seconds = _attempt_wall(path.parent, pin)
            if seconds:
                wall += seconds
                wall_known = True
    out["agent_wall_s"] = round(wall, 1) if wall_known else None
    out["request_s"] = round(request_s, 1) if request_known else None
    out["complete"] = not (skipped["oversize_files"] or skipped["files_not_read"] or skipped["walk_cut"]
                         or skipped["depth_cut"] or skipped["unreadable_files"])


def priced(scanned: dict, cfg: Config) -> dict:
    """`scanned` plus the virtual cost at each reference price. The costs are comparison figures only."""
    p, c = scanned["prompt_tokens"], scanned["completion_tokens"]
    refs = [{"name": name, "price_in": pi, "price_out": po, "cost_usd": cost(p, c, pi, po)}
            for name, pi, po in cfg.references()]
    return {**scanned, "virtual": refs, "label": LABEL}


def skipped_text(skipped: dict) -> str:
    """One plain sentence about what the scan left out, or an empty string."""
    names = {"malformed_lines": "lines that are not JSON", "no_usage": "turns with no token counts",
             "helper_turns": "turns from an outside helper", "long_lines": "lines over the size cap", "long_unverified": "lines over the size cap that do not start like an agent turn",
             "not_utf8_lines": "lines with bytes that are not UTF-8", "unreadable_files": "files that could not be read",
             "oversize_files": "files or folders over the read cap", "files_not_read": "files past the file cap", "walk_cut": "directory entries past the walk budget", "depth_cut": "folders below the depth limit",
             "links": "links that were not followed"}
    parts = [f"{skipped[k]} {names[k]}" for k in names if skipped.get(k)]
    return ("Left out: " + ", ".join(parts) + ".") if parts else ""
