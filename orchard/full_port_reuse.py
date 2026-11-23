# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""What stages 3 and 4 of a full port tested, in words the operator bundle can print (gate honesty, issue 793).

On the full-port path stage 3 copies the stage 2 port and runs the supervisor's own check on the larger mesh. Stage 4 builds no test of its
own for an entry that an earlier stage's test already covers. Both end as "pass", and the Aion run (2026-11-22) showed what a bare pass hides:
stage 3's port was the same file as stage 2's with every weight replicated, its output ids were the same as stage 2's, and stage 4 ran nothing.
All of that is true and none of it is a defect in the verdict. A reader of the bundle or the model card still needs to be told, so the bundle
names it.

Everything here is read from files the supervisor and the check wrote (`stages/N/evidence/port-check.json`, `stages/4/result.json`) and from the
port source text, through `safeio`. A record that is absent, malformed or too large gives "not known". It never gives a claim. The verdicts, the gates and
their thresholds are not touched: this module only describes.

`describe(run_dir)` returns None when the run has none of these records (another path, or an older run), so those bundles are unchanged.
"""
from __future__ import annotations

import re
from pathlib import Path

from orchard import safeio

MAX_RECORD_BYTES = 8 << 20        # a port-check record holds the reference ids and per-position numbers; the largest seen is 178 KB at 4,000 positions
MAX_SOURCE_BYTES = 4 << 20        # a generated port is far under this; a larger file is not scanned
MAX_CHIPS = 1024

_REPLICATE = re.compile(r"\bReplicate\w*Mesh\w*\s*\(")
_SHARD = re.compile(r"\bShard\w*Mesh\w*\s*\(")


def _json(path: Path, limit: int = MAX_RECORD_BYTES):
    """The parsed JSON object at `path`, or None when it is absent, unreadable, not an object, a link or a special file."""
    try:
        return safeio.read_json(path, max_bytes=limit, expect=dict)
    except safeio.SafeReadError:
        return None
    except (OSError, ValueError):
        return None


def _count(x) -> int | None:
    return x if isinstance(x, int) and not isinstance(x, bool) and 0 <= x <= MAX_CHIPS else None


def _model_shas(rec: dict | None) -> frozenset | None:
    """The sha256 of each `model` source the check hashed, or None when the record does not hold a usable list."""
    if not isinstance(rec, dict) or not isinstance(rec.get("port_sources"), list):
        return None
    out = set()
    for s in rec["port_sources"]:
        if isinstance(s, dict) and s.get("role") == "model":
            h = s.get("sha256")
            if not (isinstance(h, str) and re.fullmatch(r"[0-9a-f]{64}", h)):
                return None
            out.add(h)
    return frozenset(out) or None


def _ids(rec: dict | None):
    ids = rec.get("chip_ids") if isinstance(rec, dict) else None
    if isinstance(ids, list) and ids and all(isinstance(i, int) and not isinstance(i, bool) for i in ids):
        return ids
    return None


def _chips(rec: dict | None) -> int | None:
    c = rec.get("chips") if isinstance(rec, dict) else None
    return len(c) if isinstance(c, list) and all(isinstance(x, str) for x in c) and c else None


def _same(a, b):
    return None if a is None or b is None else a == b


def _mesh_calls(source: Path) -> tuple[int | None, int | None]:
    """(replicating places, sharding places) the port source names, or (None, None) when it cannot be read as text under the size cap.
    Comment lines are skipped. This is a count of text matches. It does not run the port."""
    try:
        data = safeio.read_bytes(source, max_bytes=MAX_SOURCE_BYTES)
        lines = data.decode("utf-8", errors="replace").splitlines()
    except (safeio.SafeReadError, OSError, ValueError):
        return None, None
    code = [ln for ln in lines if not ln.lstrip().startswith("#")]
    return sum(len(_REPLICATE.findall(ln)) for ln in code), sum(len(_SHARD.findall(ln)) for ln in code)


def describe(run_dir) -> dict | None:
    """What stage 3 re-ran and what stage 4 covered, from the run directory; None when the run has neither record."""
    run = Path(run_dir)
    s2 = _json(run / "stages" / "2" / "evidence" / "port-check.json")
    s3 = _json(run / "stages" / "3" / "evidence" / "port-check.json")
    r4 = _json(run / "stages" / "4" / "result.json")
    if s3 is None and r4 is None:
        return None
    info: dict = {}
    if s3 is not None:
        rep, shard = _mesh_calls(run / "stages" / "3" / "port" / "port_model.py")
        a, b = _model_shas(s2), _model_shas(s3)
        info["stage3"] = {"same_port": _same(a, b), "same_ids": _same(_ids(s2), _ids(s3)), "chips_before": _chips(s2), "chips_after": _chips(s3),
                          "replicated": rep, "sharded": shard,
                          "same_top1": _same(s2.get("top1_agreement") if s2 else None, s3.get("top1_agreement"))}
    if r4 is not None:
        covered, tested = [], []
        entries = r4.get("entries")
        for e in entries if isinstance(entries, list) else []:
            chips = _count(e.get("chips")) if isinstance(e, dict) else None
            if chips is None:
                continue
            by = _count(e.get("covered_by"))
            if by is not None and by in (2, 3):
                covered.append({"chips": chips, "covered_by": by})
            else:
                tested.append(chips)
        info["stage4"] = {"covered": covered, "tested": tested}
    return info


def _stages(by: list[int]) -> str:
    s = sorted(set(by))
    return "stage " + str(s[0]) if len(s) == 1 else "stages " + " and ".join(map(str, s))


def lines_for(info: dict | None, n: int) -> list[str]:
    """The sentences for stage `n` (3 or 4); empty when the records do not support any."""
    if not info:
        return []
    out: list[str] = []
    if n == 3 and isinstance(info.get("stage3"), dict):
        s = info["stage3"]
        where = "" if s["chips_after"] is None else f" on {s['chips_after']} chips" + ("" if s["chips_before"] is None else f" (stage 2 used {s['chips_before']})")
        if s["same_port"] is True:
            out.append(f"Stage 3 re-ran the port that passed stage 2{where}. The port is byte-identical to stage 2's port (the two check records hold the same sha256).")
        elif s["same_port"] is False:
            out.append(f"Stage 3 ran a port that differs from stage 2's port{where}.")
        else:
            out.append(f"Stage 3 ran a port{where}. Whether it is the same as stage 2's port is not known (a check record is missing or unreadable).")
        if s["same_ids"] is True:
            out.append("The chip's output ids equal stage 2's, token for token.")
        elif s["same_ids"] is False:
            out.append("The chip's output ids differ from stage 2's.")
        if s["replicated"] is not None and s["sharded"] is not None:
            tail = ("The port source names no sharding mapper, so by its text no weight is split over the chips."
                    if s["sharded"] == 0 and s["replicated"] > 0 else
                    f"The port source names a sharding mapper in {s['sharded']} place" + ("" if s["sharded"] == 1 else "s") + ".")
            out.append(f"The port source names a replicating mapper in {s['replicated']} place" + ("" if s["replicated"] == 1 else "s") + f" (a count of text matches). {tail}")
        else:
            out.append("How the port places its weights over the chips is not known (the source could not be read).")
        out.append("What this shows: the port runs on the larger mesh and gives the tokens it gave before. It does not show that the work is split over the chips, "
                   "or that the larger mesh is faster, or that it uses less memory per chip.")
    if n == 4 and isinstance(info.get("stage4"), dict):
        s = info["stage4"]
        if s["covered"] and not s["tested"]:
            out.append("Stage 4 ran no new test. " + " ".join(
                f"The {c['chips']}-chip entry is covered by the stage {c['covered_by']} test." for c in s["covered"]))
        else:
            for c in s["covered"]:
                out.append(f"The {c['chips']}-chip entry is covered by the stage {c['covered_by']} test.")
            for c in s["tested"]:
                out.append(f"The {c}-chip entry ran its own test.")
    return out


def section(info: dict | None) -> list[str]:
    return lines_for(info, 3) + lines_for(info, 4)


def summary_sentences(info: dict | None) -> str:
    """One or two sentences for the bundle's summary, empty when there is nothing to add."""
    if not info:
        return ""
    parts = []
    s3 = info.get("stage3")
    if isinstance(s3, dict) and s3["same_port"] is True:
        parts.append("Stage 3 re-ran stage 2's port unchanged" + ("" if s3["chips_after"] is None else f" on {s3['chips_after']} chips") + ".")
    s4 = info.get("stage4")
    if isinstance(s4, dict) and s4["covered"] and not s4["tested"]:
        parts.append(f"Stage 4 ran no new test: its entries are covered by {_stages([c['covered_by'] for c in s4['covered']])}.")
    return " ".join(parts) + (" " if parts else "")
