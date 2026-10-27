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
import json
import math
import os
import re
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
MAX_FILE_BYTES = 128 * 1024 * 1024   # a larger file is read up to this many bytes and the rest is counted as not read
MAX_TOTAL_BYTES = 1024 * 1024 * 1024
MAX_LINE_BYTES = 2 * 1024 * 1024     # a longer line is not parsed; its usage is looked for in its last bytes
TAIL_BYTES = 4096
MAX_TOKENS_PER_TURN = 10 ** 9        # a bigger number is a corrupt record

HELPER_NAMES = {"telegraph", "helper", "hosted", "outside"}
_USAGE_TAIL = re.compile(rb'"usage"\s*:\s*\{[^{}]*\}')


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
    return {"malformed_lines": 0, "no_usage": 0, "helper_turns": 0, "long_lines": 0, "not_utf8_lines": 0,
            "unreadable_files": 0, "oversize_files": 0, "files_not_read": 0, "links": 0}


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


def _agent_turn(rec: dict):
    """(prompt, completion) of an agent-log turn, or None when it has no usable usage."""
    usage = rec.get("usage")
    if not isinstance(usage, dict):
        return None
    p, c = _count(usage.get("prompt_tokens")), _count(usage.get("completion_tokens"))
    if p is None or c is None:
        return None
    return p, c


def _request(rec: dict):
    """(prompt, completion, latency) of a benchmark proxy record."""
    p, c = _count(rec.get("prompt_tokens")), _count(rec.get("completion_tokens"))
    if p is None or c is None:
        return None
    lat = rec.get("latency_s")
    lat = float(lat) if isinstance(lat, (int, float)) and not isinstance(lat, bool) and math.isfinite(lat) and lat >= 0 else 0.0
    return p, c, lat


def _lines(path: Path, skipped: dict, budget: list):
    """Yield (kind, payload) for each line of a file: ("rec", dict), ("tail", bytes) for a line over the cap. Bounded."""
    read = 0
    try:
        fh = open(path, "rb")
    except OSError:
        skipped["unreadable_files"] += 1
        return
    with fh:
        while True:
            if read >= MAX_FILE_BYTES or budget[0] <= 0:
                skipped["oversize_files"] += 1
                return
            try:
                line = fh.readline(MAX_LINE_BYTES + 1)
            except OSError:
                skipped["unreadable_files"] += 1
                return
            if not line:
                return
            read += len(line)
            budget[0] -= len(line)
            if len(line) > MAX_LINE_BYTES and not line.endswith(b"\n"):
                tail = line[-TAIL_BYTES:]
                while True:                              # drain the rest of the line, keeping only its end
                    try:
                        more = fh.readline(MAX_LINE_BYTES)
                    except OSError:
                        break
                    if not more:
                        break
                    read += len(more)
                    budget[0] -= len(more)
                    tail = (tail + more)[-TAIL_BYTES:]
                    if more.endswith(b"\n"):
                        break
                yield "tail", tail
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
    m = _USAGE_TAIL.search(tail)
    if not m:
        return None
    try:
        usage = json.loads(m.group(0).split(b":", 1)[1])
    except ValueError:
        return None
    return _agent_turn({"usage": usage})


def _find_files(root: Path, skipped: dict):
    """(path, kind) for each local log file under root, kind 'agent' or 'bench'. Links are not followed."""
    found, seen = [], 0
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        depth = len(Path(dirpath).relative_to(root).parts)
        keep = []
        for d in sorted(dirnames):
            if (Path(dirpath) / d).is_symlink():
                skipped["links"] += 1
            elif depth < MAX_DEPTH:
                keep.append(d)
        dirnames[:] = keep
        in_log = Path(dirpath).name == "log"
        for name in sorted(filenames):
            kind = "bench" if name == "requests.jsonl" else "agent" if in_log and name.endswith(".jsonl") else None
            if kind is None:
                continue
            p = Path(dirpath) / name
            if p.is_symlink() or not p.is_file():
                skipped["links"] += 1
                continue
            seen += 1
            if seen > MAX_FILES:
                skipped["files_not_read"] += 1
                continue
            found.append((p, kind))
    return found


def _attempt_wall(log_dir: Path) -> float:
    """The agent's wall seconds a kernel-probe attempt recorded beside its log directory (`attempt.json`), else 0."""
    p = log_dir.parent / "attempt.json"
    try:
        if p.is_symlink() or not p.is_file() or p.stat().st_size > 1024 * 1024:
            return 0.0
        v = json.loads(p.read_text(encoding="utf-8")).get("agent_wall_s")
    except (OSError, ValueError, AttributeError, UnicodeDecodeError):
        return 0.0
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0 else 0.0


def scan(root) -> dict:
    """Count the local turns under `root` (a directory). Raises nothing. `agent_wall_s` (from a kernel-probe attempt.json) and
    `request_s` (the benchmark proxy's request times, summed across concurrent requests, so not wall time) are None when unknown."""
    root = Path(root)
    skipped = _skipped()
    out = {"turns": 0, "prompt_tokens": 0, "completion_tokens": 0, "agent_wall_s": None, "request_s": None, "files": 0,
           "skipped": skipped, "sources": {"agent": 0, "benchmark": 0}, "complete": True}
    if root.is_symlink() or not root.is_dir():
        out["complete"] = False
        return out
    budget = [MAX_TOTAL_BYTES]
    wall, wall_known, request_s, request_known, attempts = 0.0, False, 0.0, False, set()
    for path, kind in _find_files(root, skipped):
        out["files"] += 1
        for what, payload in _lines(path, skipped, budget):
            if what == "tail":
                skipped["long_lines"] += 1
                got = _tail_usage(payload) if kind == "agent" else None
                if got is None:
                    skipped["no_usage"] += 1
                    continue
                out["turns"] += 1
                out["prompt_tokens"] += got[0]
                out["completion_tokens"] += got[1]
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
            seconds = _attempt_wall(path.parent)
            if seconds:
                wall += seconds
                wall_known = True
    out["agent_wall_s"] = round(wall, 1) if wall_known else None
    out["request_s"] = round(request_s, 1) if request_known else None
    out["complete"] = not (skipped["oversize_files"] or skipped["files_not_read"])
    return out


def priced(scanned: dict, cfg: Config) -> dict:
    """`scanned` plus the virtual cost at each reference price. The costs are comparison figures only."""
    p, c = scanned["prompt_tokens"], scanned["completion_tokens"]
    refs = [{"name": name, "price_in": pi, "price_out": po, "cost_usd": cost(p, c, pi, po)}
            for name, pi, po in cfg.references()]
    return {**scanned, "virtual": refs, "label": LABEL}


def skipped_text(skipped: dict) -> str:
    """One plain sentence about what the scan left out, or an empty string."""
    names = {"malformed_lines": "lines that are not JSON", "no_usage": "turns with no token counts",
             "helper_turns": "turns from an outside helper", "long_lines": "lines over the size cap",
             "not_utf8_lines": "lines with bytes that are not UTF-8", "unreadable_files": "files that could not be read",
             "oversize_files": "files or folders over the read cap", "files_not_read": "files past the file cap",
             "links": "links that were not followed"}
    parts = [f"{skipped[k]} {names[k]}" for k in names if skipped.get(k)]
    return ("Left out: " + ", ".join(parts) + ".") if parts else ""
