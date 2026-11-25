# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Recorded compiles behind the claims of the TT-Lang cheat sheet (issue 801).

Why. `op_cheats.verify` judged `[compiles on hardware]` and `[hardware signature only]` by finding the name and the argument names in the chip
package's Python files. `ttl.math.fill` and `ttl.math.broadcast` are defined there, and the compiler refuses them inside a kernel
(`ttl.math.fill is not available; use ttl.block.fill`): it resolves a call in a thread by its own table, which the check did not read. The
sheet taught those two names as the chip's forms and the check passed.

What. `ttlang_cheat_compiles.json` holds small probe kernels (the text is in the file) and, for each probe and each chip package, the record of
compiling it with that package's own compiler and no device (orchard/ttlang_hwcompile.py, the same record the recipes use). A probe has
`expect: "compiled"` and `covers`, the entries whose call it uses, or `expect: "refused"` and `refuses`, forms the hardware package's compiler
must refuse. scripts/verify_cheat_compiles.py writes the file. `verify` takes it with the identity of the hardware package.

The rules (`claim_problems`), judged against the records made with exactly the hardware package's compiler (version and compiler bytes):

- A `compiles on hardware` claim needs a record in which a covering probe compiled. A refusal, or a check that could not be made, is not a
  compile, and the problem names which it was.
- A `hardware signature only` claim says no compile was shown, so it needs no record. A refusal of every covering probe contradicts it. A
  compiled record does not contradict it: `understated` lists those entries so a person can raise the tag, which changes the sheet's text and
  so is a separate change.
- A form that a probe records as refused may not be an entry, whatever its tag.
- The data has to be what it says: each record is for the probe's kernel bytes (sha256), is well formed, and the hardware package is one of
  the data's targets, so a package upgrade is noticed and not read through old records.

What a record does not show: that tt-metal builds the source the compiler writes, that the buffers fit, or that the numbers are right on a
chip. It is named `compile` everywhere and never `chip`. Nothing here imports ttl or ttnn.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from orchard import safeio
from orchard import ttlang_hwcompile as hc

SCHEMA = "orchard-cheat-compiles/1"
PATH = Path(__file__).resolve().parent / "skills" / "kernel-loop-templates" / "ttlang_cheat_compiles.json"
MAX_BYTES = 800_000
MAX_PROBES = 60
MAX_KERNEL_LINES = 400
MAX_LINE = 300
MAX_NAMES = 20
MAX_TENSORS = 16
MAX_DIM = 1 << 20
EXPECTS = ("compiled", "refused")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*){0,4}")
_PROBE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,80}")


class CheatCompilesError(Exception):
    """A data file that cannot be read or has the wrong shape. The message never holds the file's content."""


# ---- reading ----------------------------------------------------------------------------------------------------------------------------

def load(path=None) -> dict:
    """The compile records through the bounded reader, checked for shape (`shape_problems`). Raises CheatCompilesError."""
    where = Path(path or PATH)
    try:
        data = safeio.read_json(where, max_bytes=MAX_BYTES, expect=dict)
    except (safeio.SafeReadError, ValueError, RecursionError) as exc:
        raise CheatCompilesError(f"{where.name}: the compile records cannot be read: {exc}") from None
    problems = shape_problems(data)
    if problems:
        raise CheatCompilesError(f"{where.name}: " + "; ".join(problems[:5]))
    return data


def probe_text(probe: dict) -> str:
    """The probe's kernel as scripts/verify_cheat_compiles.py writes it to kernel.py: its lines joined by newlines, one at the end."""
    return "\n".join(probe["kernel"]) + "\n"


def probe_sha256(probe: dict) -> str:
    return hashlib.sha256(probe_text(probe).encode("utf-8")).hexdigest()


# ---- shape ------------------------------------------------------------------------------------------------------------------------------

def _names(value, key: str, where: str, problems: list) -> None:
    if not (isinstance(value, list) and 0 < len(value) <= MAX_NAMES and all(isinstance(n, str) and _NAME.fullmatch(n) for n in value)):
        problems.append(f"{where}: {key} must list 1 to {MAX_NAMES} call names")


def _shapes(value, where: str, problems: list) -> None:
    ok = (isinstance(value, list) and 0 < len(value) <= MAX_TENSORS
          and all(isinstance(s, list) and len(s) == 2 and all(type(d) is int and 0 < d <= MAX_DIM for d in s) for s in value))
    if not ok:
        problems.append(f"{where}: shapes must list 1 to {MAX_TENSORS} pairs of positive whole numbers")


def _probe_problems(name, probe) -> list[str]:
    where = f"probe {name!r}"
    if not isinstance(probe, dict):
        return [f"{where} must be an object"]
    problems: list[str] = []
    expect = probe.get("expect")
    if expect not in EXPECTS:
        problems.append(f"{where}: expect must be one of {EXPECTS}")
    else:
        _names(probe.get("covers" if expect == "compiled" else "refuses"), "covers" if expect == "compiled" else "refuses", where, problems)
    _shapes(probe.get("shapes"), where, problems)
    kernel = probe.get("kernel")
    if not (isinstance(kernel, list) and 0 < len(kernel) <= MAX_KERNEL_LINES and all(isinstance(ln, str) and len(ln) <= MAX_LINE for ln in kernel)):
        problems.append(f"{where}: kernel must be a list of 1 to {MAX_KERNEL_LINES} lines of text")
        kernel = None
    sha = probe.get("kernel_sha256")
    if not (isinstance(sha, str) and _HEX64.fullmatch(sha)):
        problems.append(f"{where}: kernel_sha256 must be 64 hex digits")
    elif kernel is not None and sha != probe_sha256(probe):
        problems.append(f"{where}: kernel_sha256 does not match the kernel text (the text was edited after it was compiled)")
    compiles = probe.get("compile")
    if not (isinstance(compiles, list) and len(compiles) <= 8):
        problems.append(f"{where}: compile must be a list of at most 8 records")
        return problems
    for i, item in enumerate(compiles):
        if not (isinstance(item, dict) and isinstance(item.get("date"), str) and isinstance(item.get("record"), dict)):
            problems.append(f"{where}: compile[{i}] needs a date and a record")
            continue
        record = item["record"]
        for p in hc.record_problems(record):
            problems.append(f"{where}: compile[{i}]: {p}")
        theirs = record.get("kernel_sha256")
        if isinstance(sha, str) and isinstance(theirs, str) and theirs != sha:
            problems.append(f"{where}: compile[{i}]: the record is for another kernel than the probe's text")
    return problems


def shape_problems(data) -> list[str]:
    """What is wrong with the shape of a data file already read: its schema, the targets, and every probe with its records."""
    if not isinstance(data, dict):
        return ["the compile records must be an object"]
    problems = []
    if data.get("schema") != SCHEMA:
        problems.append(f"the compile records must have schema {SCHEMA!r}")
    if not isinstance(data.get("date"), str):
        problems.append("the compile records need a date")
    targets = data.get("targets")
    if not (isinstance(targets, list) and targets):
        problems.append("targets must list the chip packages the records were made with")
    else:
        for t in targets:
            problems += [f"targets: {p}" for p in hc.identity_problems(t)]
    probes = data.get("probes")
    if not (isinstance(probes, dict) and 0 < len(probes) <= MAX_PROBES):
        problems.append(f"probes must be an object of 1 to {MAX_PROBES} probes")
        return problems
    for name, probe in probes.items():
        if not (isinstance(name, str) and _PROBE.fullmatch(name)):
            problems.append(f"probe name {str(name)[:40]!r} is not a plain name")
            continue
        problems += _probe_problems(name, probe)
    return problems


# ---- the records of one hardware package -----------------------------------------------------------------------------------------------

def usable_records(probe: dict, target: dict) -> list[dict]:
    """The probe's compile records made with exactly `target`'s compiler and well formed. A malformed record is skipped here (the data's
    shape problems name it), so it can never support a claim."""
    out = []
    for item in probe.get("compile", []):
        record = item.get("record") if isinstance(item, dict) else None
        if isinstance(record, dict) and not hc.record_problems(record) and hc.same_target(record.get("target"), target):
            out.append(record)
    return out


def refused_forms(data: dict, target: dict) -> dict:
    """{form: the compiler's first line} for every form a `refused` probe lists that `target`'s compiler refused (category compile_error)."""
    out = {}
    for probe in data.get("probes", {}).values():
        if probe.get("expect") != "refused":
            continue
        for record in usable_records(probe, target):
            if record["category"] == "compile_error":
                for form in probe["refuses"]:
                    if form in record["first_line"]:                # the refusal names the form; a failure for another reason does not count
                        out[form] = record["first_line"]
    return out


def _entries(sheet: dict):
    for section in sheet.get("sections", []):
        yield from section.get("entries", [])


def _covering(data: dict, target: dict) -> dict:
    """{entry name: [records of the probes that cover it, for `target`]}."""
    out: dict = {}
    for probe in data["probes"].values():
        if probe["expect"] != "compiled":
            continue
        records = usable_records(probe, target)
        for name in probe["covers"]:
            out.setdefault(name, []).extend(records)
    return out


def claim_problems(sheet: dict, data: dict, hardware_target) -> list[str]:
    """Problems with the sheet's chip claims judged by the recorded compiles of the hardware package `hardware_target` (an identity as
    ttlang_hwcompile.package_identity returns it). An empty list means every `compiles on hardware` claim has a compiled record, no claim is
    contradicted by a refusal, and no entry is a form the hardware compiler refuses."""
    problems = shape_problems(data)
    if problems:
        return problems
    ident = hc.identity_problems(hardware_target)
    if ident:
        return [f"the hardware package's identity is not usable ({ident[0]}), so no claim was checked against a compile"]
    if not any(hc.same_target(hardware_target, t) for t in data["targets"]):
        return [f"the compile records were not made with this hardware package ({hc.target_label(hardware_target)}, compiler "
                f"{hardware_target['compiler_sha256'][:12]}): run scripts/verify_cheat_compiles.py to record them"]
    refused = refused_forms(data, hardware_target)
    covering = _covering(data, hardware_target)
    entry_names = {e.get("name") for e in _entries(sheet)}
    for pname, probe in data["probes"].items():
        if probe["expect"] == "compiled":
            problems += [f"probe {pname!r} covers {n}, which the sheet has no entry for" for n in probe["covers"] if n not in entry_names]
        else:
            verdicts = usable_records(probe, hardware_target)
            if any(r["category"] == "compiled" for r in verdicts):
                problems.append(f"probe {pname!r} was expected to be refused by the hardware package and it compiled")
            elif any(r["category"] == "compile_error" for r in verdicts):
                if not all(form in refused for form in probe["refuses"]):
                    problems.append(f"probe {pname!r} failed to compile with the hardware package, but the compiler's line does not name "
                                    f"{', '.join(probe['refuses'])}, so the failure is not a refusal of it ({verdicts[0]['first_line']})")
            else:
                got = verdicts[0]["category"] if verdicts else "no record"
                problems.append(f"probe {pname!r} was expected to be refused and the hardware package's check gave no verdict ({got})")
    for entry in _entries(sheet):
        name, status = entry.get("name"), entry.get("status")
        if name in refused:
            problems.append(f"{name}: the hardware package's compiler refuses this form ({refused[name]}); remove the entry or write the form "
                            f"the message names")
            continue
        records = covering.get(name, [])
        compiled = [r for r in records if r["category"] == "compiled"]
        refusals = [r for r in records if r["category"] == "compile_error"]
        if status == "compiles on hardware" and not compiled:
            if refusals:
                problems.append(f"{name}: status 'compiles on hardware' but the hardware package's compiler refused every kernel that uses it "
                                f"({refusals[0]['first_line']})")
            else:
                problems.append(f"{name}: status 'compiles on hardware' but no compile result with the hardware package covers it (the name "
                                f"and arguments found in the package's files do not show that the compiler takes the call)")
        elif status == "hardware signature only" and refusals and not compiled:
            problems.append(f"{name}: status 'hardware signature only' but the hardware package's compiler refused a kernel that uses it "
                            f"({refusals[0]['first_line']})")
    return problems


def understated(sheet: dict, data: dict, hardware_target) -> list[str]:
    """The `hardware signature only` entries that a compiled record of the hardware package covers: their tag could say `compiles on hardware`.
    Empty when the data is malformed or was not made with this hardware package (`claim_problems` says so)."""
    if shape_problems(data) or hc.identity_problems(hardware_target) or not any(hc.same_target(hardware_target, t) for t in data["targets"]):
        return []
    covering = _covering(data, hardware_target)
    return [e["name"] for e in _entries(sheet)
            if e.get("status") == "hardware signature only" and any(r["category"] == "compiled" for r in covering.get(e["name"], []))]
