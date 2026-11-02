# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Turn a raw op trace into an inventory: which of a model's ops ttnn already has, which it can build, which are missing.

Statuses, and what each claims:

    mapped      the mapping table names a ttnn op and the recorded dump for the installed ttnn contains that name.
                This is a name match. It does not say the op supports the shape, dtype or layout the model needs.
    composable  a decomposition exists and every leaf is `mapped`.
    verified    no table row, but `tt-orchard inventory verify-candidates` ran the ttnn op against a torch reference on
                the ttsim simulator with one small shape and it agreed (orchard/data/verified_mappings.json). It is
                not `mapped`: a person still has to add the table row, and the simulator is not hardware.
    candidate   no table row, but orchard/opvariants.py found up to three ttnn names that might do the job. A guess
                about names; `candidates` on the row says why each was proposed. Not `mapped`.
    unmapped    none of the above.
    ignored     the op is listed in ignored_ops.json: pure metadata that needs no ttnn op. It makes no ttnn claim,
                so it is `ignored` with or without a dump.
    host        the op is listed in op_notes.json with `host: true`: it runs on the host by design (a constant built
                once, preparation of an input, a lookup in a table too big for the device). The entry carries a reason
                and the evidence for it. It makes no ttnn claim, so it is `host` with or without a dump. A table row,
                a decomposition, a verified mapping or `ignored` takes precedence.
    unchecked   there is no dump for the installed ttnn version, so nothing is claimed.

op_notes.json also holds notes for ops of any status (why an op is `composable`, or why an expert-loop op stays
`unmapped`). A note never changes a status unless it says `host: true`.

A trace that carries an ordered op list (`sequence`) is also searched for groups of ops that one ttnn op does at once
(orchard/oppatterns.py). Those are listed apart, in `patterns`, with status `pattern` (or `pattern-verified` after
verify-candidates). The ops inside a group keep their own statuses.

The dump comes from `tt-orchard inventory refresh-ttnn` (orchard/ttnn_dump.py), which imports ttnn under a lease,
because ttnn's op names cannot be read without importing it. A dump for another ttnn version than the installed one is
not used.

Standard library only. The torch side is orchard/skills/ops-inventory-templates/ops_inventory.py.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from orchard import knownwrong, opvariants, oppatterns, safeio

MAX_DATA_BYTES = 16 * 1024 * 1024       # choice: a dump or a table is tens of kilobytes

DATA_DIR = Path(__file__).resolve().parent / "data"
TEMPLATE = Path(__file__).resolve().parent / "skills" / "ops-inventory-templates" / "ops_inventory.py"
SCHEMA = 1
STATUSES = ("mapped", "composable", "verified", "candidate", "unmapped", "unchecked", "ignored", "host")
VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")
LABEL = {
    "traced": "measured: the op names, counts and one sample shape per op come from torch.export on the meta device",
    "status": "table plus dump: `mapped` is a name match against the recorded ttnn dump and does not say the op "
              "supports the model's shape, dtype or layout",
    "scaled_count": "estimate: ops inside layers are multiplied by layers in the config / layers traced",
    "candidate": "guess: a ttnn name that resembles the aten op's name or its synonym; nothing says it does the same job",
    "pattern": "recognised from the order and data flow of the traced ops; the ttnn op exists in the dump, and nothing "
               "says it takes this model's shapes, dtype or layout",
    "verified": "one small shape on the ttsim simulator against a torch reference; not run on a chip",
    "host": "a table entry in op_notes.json says the op runs on the host by design; the entry gives the reason and the "
            "evidence, and nothing was run to time the host path",
}


def _read(path):
    """The parsed JSON file at `path`. A file that is missing, is not a regular file, is over MAX_DATA_BYTES or is not strict JSON is a
    ValueError, the documented error of the loaders below (issues 135 to 139)."""
    try:
        return safeio.read_json(safeio.resolved(path), max_bytes=MAX_DATA_BYTES)
    except safeio.SafeReadError as exc:
        raise ValueError(f"{path}: {exc.reason}") from exc


def load_table(path) -> dict:
    """Read a mapping table: aten op name to {"ttnn": name}, or (for decompositions) to a list of aten op names."""
    data = _read(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    for key, entry in data.items():
        if isinstance(entry, dict):
            if not isinstance(entry.get("ttnn"), str):
                raise ValueError(f"{path}: {key!r} has no string `ttnn`")
        elif not (isinstance(entry, list) and all(isinstance(x, str) for x in entry)):
            raise ValueError(f"{path}: {key!r} must be {{\"ttnn\": name}} or a list of aten op names")
    return data


def load_ignored(path) -> frozenset:
    """Read the list of aten ops that are metadata only."""
    data = _read(path)
    if not (isinstance(data, list) and all(isinstance(x, str) for x in data)):
        raise ValueError(f"{path}: expected a JSON list of aten op names")
    return frozenset(data)


NOTE_KEYS = frozenset({"reason", "evidence", "host"})


def load_notes(path) -> dict:
    """Read the op notes table: aten op name to {"reason": str, "evidence": str, "host": bool (optional)}.
    Every entry needs a non-empty reason and a non-empty evidence string; `host` must be a boolean when present and
    defaults to False; any other key is refused, as is an empty op name. Returns {op: {reason, evidence, host}}."""
    data = _read(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    out = {}
    for key, entry in data.items():
        if not key.strip():
            raise ValueError(f"{path}: an entry has an empty op name")
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: {key!r} must be an object with `reason` and `evidence`")
        extra = sorted(set(entry) - NOTE_KEYS)
        if extra:
            raise ValueError(f"{path}: {key!r} has unknown keys {extra}")
        for field in ("reason", "evidence"):
            v = entry.get(field)
            if not isinstance(v, str) or not v.strip():
                raise ValueError(f"{path}: {key!r} needs a non-empty string `{field}`")
        host = entry.get("host", False)
        if not isinstance(host, bool):
            raise ValueError(f"{path}: {key!r} has a `host` that is not true or false")
        out[key] = {"reason": entry["reason"], "evidence": entry["evidence"], "host": host}
    return out


BUILD_DATE_RE = re.compile(r"^(\d{8})(?:-[0-9a-f]{7,40})?$")


def build_date(build) -> str | None:
    """The YYYYMMDD part of a build id (`20261030` or `20261030-<commit>`), or None. A commit-only id has no date."""
    m = BUILD_DATE_RE.match(build) if isinstance(build, str) else None
    return m.group(1) if m else None


def _read_dump(p, version):
    """One dump file for `version`, or None when it is missing, unreadable, for another version or has no name list."""
    if not p.is_file():
        return None
    try:
        d = safeio.read_json(safeio.resolved(p), max_bytes=MAX_DATA_BYTES, expect=dict)
    except safeio.SafeReadError:
        return None
    return d if d.get("ttnn_version") == version and isinstance(d.get("names"), list) else None


def load_dump(version, data_dir=DATA_DIR, build=None):
    """The recorded dump for this ttnn version, or None. A version that is not a plain token is refused, so it cannot name a
    file outside data_dir.

    A build can change the op list while it keeps its version (issue 339), so a version can have several dumps: the legacy
    `ttnn_ops-<version>.json` and `ttnn_ops-<version>-build-<id>.json` for each build recorded. `build` is the installed build's
    date (YYYYMMDD) when known: a dump with that build date wins. Otherwise the newest dump wins: the dump with the latest build
    date, and a legacy dump with no build counts as the oldest. build() refuses a dump whose build date differs from the
    installed one."""
    if not isinstance(version, str) or not VERSION_RE.match(version):
        return None
    root = Path(data_dir)
    found = [d for d in (_read_dump(p, version) for p in [root / f"ttnn_ops-{version}.json",
                                                          *sorted(root.glob(f"ttnn_ops-{version}-build-*.json"))]) if d]
    if not found:
        return None
    if build is not None:
        for d in found:
            if build_date(d.get("build")) == build:
                return d
    # Latest build date first; a dump that names any build (a commit-only id has no date) beats a legacy dump with none.
    return max(found, key=lambda d: (build_date(d.get("build")) or "", bool(d.get("build"))))


def load_verified(path=None) -> dict:
    """{aten op or `pattern:<kind>`: entry} from verified_mappings.json, or {} when there is none or it is unusable."""
    p = Path(path) if path else DATA_DIR / "verified_mappings.json"
    try:
        d = safeio.read_json(safeio.resolved(p), max_bytes=MAX_DATA_BYTES)
    except safeio.SafeReadError:
        return {}
    ents = d.get("entries") if isinstance(d, dict) else None
    return {k: v for k, v in ents.items() if isinstance(v, dict) and isinstance(v.get("ttnn"), str)} \
        if isinstance(ents, dict) else {}


def status_of(op, mapping, decomps, names, seen=(), ignored=frozenset(), verified=None, host=None):
    """(status, ttnn name, leaves). `names` is the set of names in the dump, or None for no usable dump.
    `ignored` is the set of metadata-only ops; they are checked first because they make no ttnn claim.
    `verified` is {op: entry} from verified_mappings.json; an entry counts only while its ttnn name is in the dump.
    `host` is {op: note} from load_notes; only entries with `host: true` count. A host op makes no ttnn claim, so it is
    `host` with or without a dump; a table row whose name is in the dump, a decomposition and a verified mapping take
    precedence."""
    if op in ignored:
        return "ignored", None, []
    entry = mapping.get(op)
    is_host = bool((host or {}).get(op, {}).get("host"))
    if names is None:
        if is_host and entry is None:
            return "host", None, []
        return "unchecked", (entry or {}).get("ttnn"), []
    if entry and entry["ttnn"] in names:
        return "mapped", entry["ttnn"], []
    parts = decomps.get(op)
    if parts and op not in seen:
        leaves, ok = [], True
        for p in parts:
            s, _, sub = status_of(p, mapping, decomps, names, seen + (op,), ignored)
            if s == "mapped":
                leaves.append(p)
            elif s == "composable":
                leaves.extend(sub)
            elif s == "ignored":
                pass                      # metadata only: nothing to build
            else:
                ok = False
        if ok:
            return "composable", None, leaves
    v = (verified or {}).get(op)
    if v and v["ttnn"] in names:
        return "verified", v["ttnn"], []
    if is_host:
        return "host", None, []
    return "unmapped", (entry or {}).get("ttnn"), []


def build(raw, *, mapping, decomps, dump, installed_ttnn=None, against=None, ignored=frozenset(), verified=None,
          variants=True, installed_build=None, notes=None) -> dict:
    """The inventory dict for one raw trace. `against` is another inventory (the base model's) to compare with.
    `installed_build` is the installed build's date (YYYYMMDD) when known; a dump from another build date is not used.
    `variants` turns on the search for candidate ttnn names for ops with no table row.
    `notes` is {op: note} from load_notes: a note with `host: true` makes the op `host`, and every note is attached to
    its row as `note` ({reason, evidence})."""
    notes = notes or {}
    inv = {"schema": SCHEMA, "model_dir": raw.get("model_dir"), "architectures": raw.get("architectures", []),
           "traced_class": raw.get("traced_class"), "torch": raw.get("torch"), "transformers": raw.get("transformers"),
           "traced": {"graph": "prefill", "seq_len": raw.get("seq_len"), "layers_in_config": raw.get("layers_in_config"),
                      "layers_traced": raw.get("layers_traced"), "note": raw.get("note"),
                      # optional in a raw trace (older traces lack them); python-level graph nodes are not ops
                      "python_nodes": raw.get("python_nodes") or {},
                      "higher_order_nodes": raw.get("higher_order_nodes") or {},
                      # modules replaced by a static stand-in (--stub-data-dependent): their ops are a guide, not the real graph
                      "stubbed": raw.get("stubbed") or [],
                      # a qwen4_exp config with dense layers was built with the provisional dense-layer shim (orchard/skills/reference-gate-templates/qwen4_dense.py)
                      "dense_layers": raw.get("dense_layers"), "dense_note": raw.get("dense_note")},
           "failures": raw.get("failures", []), "label": LABEL,
           "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    if "skipped" in raw:
        inv.update(skipped=raw["skipped"], ops=[], summary={}, ttnn_dump=None, ttnn_dump_note=None, scaling=None)
        return inv
    names, note = None, None
    if dump is None:
        note = "no recorded dump for the installed ttnn; nothing is claimed as mapped"
    elif installed_ttnn is None:
        note = "the installed ttnn version is unknown, so the recorded dump is not used"
    elif dump.get("ttnn_version") != installed_ttnn:
        note = f"the recorded dump is for ttnn {dump.get('ttnn_version')} and ttnn {installed_ttnn} is installed; not used"
    elif (installed_build and build_date(dump.get("build")) and build_date(dump.get("build")) != installed_build):
        note = (f"the recorded dump is for the ttnn {dump['ttnn_version']} build of {build_date(dump['build'])} and the installed "
                f"build is from {installed_build}; not used")
    else:
        names = set(dump["names"])
    inv["ttnn_dump"] = {"ttnn_version": dump["ttnn_version"], "sha256": dump.get("sha256")} if names is not None else None
    inv["ttnn_dump_note"] = note
    layers, traced = raw.get("layers_in_config"), raw.get("layers_traced")
    factor = layers / traced if layers and traced else None
    if not factor:
        inv["scaling"] = "the config has no recognisable layer count: counts are not scaled"
    elif layers == traced:
        inv["scaling"] = "all layers traced: no scaling needed"
    else:
        inv["scaling"] = f"in-layer ops scaled by {layers}/{traced}"
    base_ops = {o["op"] for o in against["ops"]} if against else None
    index = opvariants.Index(names) if (variants and names is not None) else None
    rows, distinct, scaled_total = [], {s: 0 for s in STATUSES}, {s: 0 for s in STATUSES}
    for o in raw["ops"]:
        status, ttnn_name, leaves = status_of(o["op"], mapping, decomps, names, ignored=ignored, verified=verified, host=notes)
        cands = opvariants.candidates(o["op"], index) if (index is not None and status == "unmapped") else []
        if cands:
            status = "candidate"
        inl = o.get("count_in_layers", 0)
        scaled = round(inl * factor + max(o["count"] - inl, 0)) if factor else o["count"]
        row = {"op": o["op"], "count": o["count"], "scaled_count": scaled, "status": status, "ttnn": ttnn_name,
               "leaves": leaves, "sample": o.get("sample"), "modules": o.get("modules", []), "candidates": cands}
        if o["op"] in notes:
            row["note"] = {"reason": notes[o["op"]]["reason"], "evidence": notes[o["op"]]["evidence"]}
        if status == "verified":
            row["verified"] = {k: v for k, v in (verified or {}).get(o["op"], {}).items() if k != "ttnn"}
        if base_ops is not None:
            row["in_base"] = o["op"] in base_ops
        rows.append(row)
        distinct[status] += 1
        scaled_total[status] += scaled
    inv["ops"] = rows
    pat = oppatterns.recognise(raw.get("sequence"), names, layers) if names is not None else {"found": [], "unavailable": []}
    found, unavailable = pat["found"], pat["unavailable"]
    for f in found:
        v = (verified or {}).get("pattern:" + f["pattern"])
        f["status"] = "pattern-verified" if v and v["ttnn"] == f["ttnn"] else "pattern"
        if f["status"] == "pattern-verified":
            f["verified"] = {k: x for k, x in v.items() if k != "ttnn"}
    inv["patterns"], inv["patterns_unavailable"] = found, unavailable
    # ttnn calls that returned a wrong result with no error on a probed build (orchard/knownwrong.py). The flag does not depend on
    # the dump: it comes from the arguments the trace recorded.
    inv["known_wrong"] = knownwrong.flag_sequence(raw.get("sequence"))
    inv["summary"] = {"distinct": distinct, "scaled_total": scaled_total, "patterns": len(found)}
    return inv


def render_md(inv) -> str:
    lines = [f"# Op inventory: {inv.get('traced_class') or inv.get('architectures')}", ""]
    if inv.get("skipped"):
        return "\n".join(lines + [f"Skipped: {inv['skipped']}", ""])
    d = inv.get("ttnn_dump")
    lines += [f"Model directory: `{inv['model_dir']}`. transformers {inv.get('transformers')}, torch {inv.get('torch')}.",
              f"Traced: {inv['traced']['note']}. Layers in config {inv['traced']['layers_in_config']}, "
              f"traced {inv['traced']['layers_traced']}. {inv['scaling']}.", ""]
    if d:
        lines.append(f"`mapped` means the name exists in the dump of ttnn {d['ttnn_version']}. It does not say the op "
                     "supports the shape, dtype or layout this model needs.")
    else:
        lines.append(f"Nothing is claimed as mapped: {inv['ttnn_dump_note']}.")
    lines += ["", "Ignored ops are metadata only (views of a tensor that is already there) and need no ttnn op. "
                  "`host` ops run on the host by design (a constant built once, preparation of an input, a table lookup); "
                  "the reason and the evidence are in the notes below the op table, and nothing was run to time the host path.",
              "", "| status | distinct ops |", "| --- | --- |"]
    lines += [f"| {s} | {n} |" for s, n in inv["summary"]["distinct"].items()]
    for st in inv["traced"].get("stubbed") or []:
        lines += ["", f"Stand-in: {st['count']} `{st['module']}` module(s) were replaced by a static version that keeps the "
                      f"op mix and ignores which positions each query can see. Their ops are a guide to what the layer "
                      f"needs and the graph is not the real graph ({st['note']})."]
    if inv["traced"].get("dense_note"):
        dense = inv["traced"].get("dense_layers") or []
        lines += ["", f"Dense layers: {len(dense)} of the traced layers use a plain MLP, built by a shim over transformers' MoE-only qwen4_exp "
                      f"class ({inv['traced']['dense_note']})."]
    py_nodes = sum((inv["traced"].get("python_nodes") or {}).values())
    if py_nodes:
        lines += ["", f"{py_nodes} python-level nodes (getitem and similar) are not ops and are not counted."]
    hop_nodes = sum((inv["traced"].get("higher_order_nodes") or {}).values())
    if hop_nodes:
        lines += ["", f"The bodies of {hop_nodes} higher-order operations were traversed; their ops are counted."]
    if inv["failures"]:
        lines += ["", "Trace failures:"] + [f"- {f.get('stage')}: {f.get('error')}" for f in inv["failures"]]
    lines += ["", "`candidate` ops have no table row; the names in the last column are guesses made from the op's name "
                  "(why in brackets). `verified` ops agreed with a torch reference on the ttsim simulator at one small "
                  "shape. Neither is `mapped`.",
              "", "| op | count | scaled | status | ttnn |", "| --- | --- | --- | --- | --- |"]
    order = {s: i for i, s in enumerate(("unmapped", "unchecked", "candidate", "verified", "composable", "mapped",
                                         "host", "ignored"))}
    for r in sorted(inv["ops"], key=lambda r: (order[r["status"]], -r["scaled_count"], r["op"])):
        shown = r["ttnn"] or ""
        if r["status"] == "candidate":
            shown = ", ".join(f"{c['ttnn']} ({c['why']})" for c in r.get("candidates") or [])
        lines.append(f"| `{r['op']}` | {r['count']} | {r['scaled_count']} | {r['status']} | {shown} |")
    noted = [r for r in inv["ops"] if r.get("note")]
    if noted:
        lines += ["", "Notes on ops (`op_notes.json`): why an op has its status, and the evidence.", "",
                  "| op | status | reason | evidence |", "| --- | --- | --- | --- |"]
        for r in sorted(noted, key=lambda r: (order[r["status"]], r["op"])):
            lines.append(f"| `{r['op']}` | {r['status']} | {r['note']['reason']} | {r['note']['evidence']} |")
    pats = inv.get("patterns") or []
    if pats or inv.get("patterns_unavailable"):
        lines += ["", "Groups of ops that one ttnn op does at once, recognised from the order and data flow of the trace "
                      "(`pattern`; `pattern-verified` after the simulator check). The ops inside a group keep their own "
                      "statuses above.", "", "| group | ttnn | status | ops in a group | found | scaled |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for f in pats:
            lines.append(f"| {f['pattern']} | {f['ttnn']} | {f['status']} | {f['nodes']} | {f['occurrences']} | "
                         f"{f['scaled_count']} |")
        for u in inv.get("patterns_unavailable") or []:
            lines.append(f"| {u['pattern']} | none of {', '.join(u['wanted'])} is in the dump | not claimed | | "
                         f"{u['occurrences']} | |")
    for f in pats:
        if f.get("note"):
            lines += ["", f"`{f['pattern']}`: {f['note']}"]
    kw = inv.get("known_wrong") or []
    if kw:
        lines += ["", "Known wrong on the build that was probed. These traced calls match a ttnn call that returned a wrong "
                      "result with no error. A `mapped` status above does not clear them.", "",
                  "| traced op | module | dilation | groups | issue | workaround |", "| --- | --- | --- | --- | --- | --- |"]
        for f in kw:
            lines.append(f"| `{f['op']}` | {f.get('module') or ''} | {f['dilation']} | {f['groups']} | {f['issue']} | "
                         f"{f['workaround']} |")
    return "\n".join(lines) + "\n"


def write_outputs(inv, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    j, m = out / "ops-inventory.json", out / "ops-inventory.md"
    j.write_text(json.dumps(inv, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    m.write_text(render_md(inv), encoding="utf-8")
    return j, m


def _real_run(argv, timeout):
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as exc:
        return 127, "", f"{type(exc).__name__}: {exc}"
    return done.returncode, done.stdout, done.stderr


def build_id(pkg_dir, commit=None):
    """An id for the compiled ttnn in `pkg_dir`: the UTC date of the newest `_ttnn*.so` there (YYYYMMDD), then `-<first 12 hex of
    commit>` when a commit is known. A commit alone gives just the 12 hex. None when neither is known. The version string does not
    tell builds apart (issue 339), so a dump is keyed by this id. This function's source is copied into the script that runs
    under the lease (orchard/ttnn_dump.py), so it must stay self-contained."""
    import datetime
    import glob
    import os
    import re
    date = None
    for p in glob.glob(os.path.join(str(pkg_dir), "_ttnn*.so")):
        try:
            d = datetime.datetime.fromtimestamp(os.path.getmtime(p), datetime.timezone.utc).strftime("%Y%m%d")
        except OSError:
            continue
        date = d if date is None or d > date else date
    short = commit[:12].lower() if isinstance(commit, str) and re.fullmatch(r"[0-9a-fA-F]{7,40}", commit) else None
    return "-".join(x for x in (date, short) if x) or None


def installed_ttnn_build(run, python):
    """The installed ttnn's build date (YYYYMMDD), or None. It reads the date of the compiled library next to the package and
    never imports ttnn, which would open the device."""
    if not python:
        return None
    import inspect
    code = ("import importlib.util, os\n" + inspect.getsource(build_id) +
            "s = importlib.util.find_spec('ttnn')\nprint((build_id(os.path.dirname(s.origin)) or '') if s and s.origin else '')")
    rc, out, _ = run([str(python), "-c", code], 60)
    return build_date(out.strip()) if rc == 0 else None


def installed_ttnn_version(run, python):
    """The installed ttnn's version, from package metadata. ttnn itself is not imported: that would open the device."""
    if not python:
        return None
    rc, out, _ = run([str(python), "-c", "import importlib.metadata as m; print(m.version('ttnn'))"], 60)
    return out.strip() or None if rc == 0 else None


def run_inventory(model_dir, *, python, ttnn_python, out_dir, layers=2, seq=8, cls=None, against=None,
                  run=_real_run, dump_dir=DATA_DIR, say=print, verified_file=None, stub_data_dependent=False) -> int:
    """Trace, map, write. 0 on success (a skipped model is a success), 2 when the template could not run or a
    file it needs is unreadable. The tables always come from the package; `dump_dir` only says where the ttnn dump is.
    `verified_file` is a verified_mappings.json to read instead of the package's."""
    base = None
    if against:
        try:
            base = json.loads(Path(against).read_text(encoding="utf-8"))
            if not isinstance(base, dict) or not isinstance(base.get("ops"), list):
                raise ValueError("no `ops` list")
        except (OSError, ValueError) as exc:
            say(f"cannot use the --against inventory {against}: {exc}")
            return 2
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / "raw-ops.json"
    argv = [str(python), str(TEMPLATE), "--model-dir", str(model_dir), "--out", str(raw_path),
            "--layers", str(layers), "--seq", str(seq)] + (["--class", cls] if cls else []) \
        + (["--stub-data-dependent"] if stub_data_dependent else [])
    rc, stdout, stderr = run(argv, 1800)
    if rc not in (0, 3) or not raw_path.is_file():
        say(f"the trace did not run (exit {rc}): {(stderr or stdout).strip()[-300:]}")
        return 2
    try:
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("not a JSON object")
    except (OSError, ValueError) as exc:
        say(f"raw-ops.json is unreadable: {exc}")
        return 2
    version = installed_ttnn_version(run, ttnn_python)
    installed_build = installed_ttnn_build(run, ttnn_python)
    inv = build(raw, mapping=load_table(DATA_DIR / "aten_to_ttnn.json"),
                decomps=load_table(DATA_DIR / "decompositions.json"), dump=load_dump(version, dump_dir, build=installed_build),
                installed_ttnn=version, against=base, ignored=load_ignored(DATA_DIR / "ignored_ops.json"),
                verified=load_verified(verified_file), installed_build=installed_build,
                notes=load_notes(DATA_DIR / "op_notes.json"))
    j, m = write_outputs(inv, out)
    say(f"wrote {j} and {m}")
    if inv.get("skipped"):
        say(f"skipped: {inv['skipped']}")
    else:
        say("distinct ops by status: " + ", ".join(f"{s} {n}" for s, n in inv["summary"]["distinct"].items()))
        if inv["patterns"]:
            say("op groups recognised: " + ", ".join(f"{f['pattern']} -> {f['ttnn']} ({f['scaled_count']})"
                                                     for f in inv["patterns"]))
        for f in inv.get("known_wrong") or []:
            say(f"KNOWN WRONG: {f['op']} (dilation {f['dilation']}, groups {f['groups']}) matches ttnn.{f['ttnn']}, which returned "
                f"a wrong result with no error on {f['build']} (issue {f['issue']}). Workaround: {f['workaround']}")
        if inv["summary"]["distinct"]["candidate"]:
            say("candidates are guesses and are not mapped; `tt-orchard inventory verify-candidates` tests them "
                "on the simulator")
    return 0


UNMAPPED_LIST_MAX = 10


def record_in_delta(stage_dir) -> bool:
    """Merge a short summary of `ops-inventory.json` into `delta.json`, both in `stage_dir`, under the key `inventory`.

    The summary holds the traced class, the skip reason (or None), the distinct-op counts by status, up to ten
    `unmapped` op names (most frequent first, by scaled count), the same for `candidate` ops with their proposed ttnn
    names (labelled guesses, never `mapped`), the recognised op groups and the ttnn dump the statuses were checked
    against.
    A skipped inventory records the reason and no counts. Every other key of delta.json is left as it was. The file
    is replaced through a temporary file in the same directory, so a reader never sees half a file. Returns False and
    changes nothing when either file is missing or is not a JSON object.
    """
    sd = Path(stage_dir)
    try:
        inv = json.loads((sd / "ops-inventory.json").read_text(encoding="utf-8"))
        delta = json.loads((sd / "delta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(inv, dict) or not isinstance(delta, dict):
        return False
    rec = {"traced_class": inv.get("traced_class"), "skipped": inv.get("skipped") or None}
    if not rec["skipped"]:
        rows = [o for o in inv.get("ops") or [] if isinstance(o, dict) and o.get("status") == "unmapped"]
        rows.sort(key=lambda o: -(o.get("scaled_count") or 0))
        rec["distinct"] = (inv.get("summary") or {}).get("distinct")
        rec["unmapped_ops"] = [o.get("op") for o in rows[:UNMAPPED_LIST_MAX]]
        cand = [o for o in inv.get("ops") or [] if isinstance(o, dict) and o.get("status") == "candidate"]
        cand.sort(key=lambda o: -(o.get("scaled_count") or 0))
        rec["candidate_ops"] = [{"op": o.get("op"),
                                 "candidates": [c.get("ttnn") for c in (o.get("candidates") if isinstance(
                                     o.get("candidates"), list) else []) if isinstance(c, dict) and isinstance(c.get("ttnn"), str)]}
                                for o in cand[:UNMAPPED_LIST_MAX]]
        rec["patterns"] = [{"pattern": f.get("pattern"), "ttnn": f.get("ttnn"), "status": f.get("status"),
                            "scaled_count": f.get("scaled_count")}
                           for f in (inv.get("patterns") if isinstance(inv.get("patterns"), list) else [])
                           if isinstance(f, dict)]
        kw = [{k: f.get(k) for k in ("op", "ttnn", "dilation", "groups", "issue")}
              for f in (inv.get("known_wrong") if isinstance(inv.get("known_wrong"), list) else []) if isinstance(f, dict)]
        if kw:                                  # present only when something matched, so an inventory with none keeps its old shape
            rec["known_wrong"] = kw
    rec["ttnn_dump"] = inv.get("ttnn_dump")
    delta["inventory"] = rec
    tmp = sd / f".delta.json.{os.getpid()}.tmp"
    try:
        tmp.write_text(json.dumps(delta, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, sd / "delta.json")
    except OSError:
        tmp.unlink(missing_ok=True)
        return False
    return True


def _real_home() -> Path:
    """The operator's home from the passwd entry. `tt-orchard inventory` uses $HOME, which is the same home when the
    operator runs it; the stage 0 agent runs this module in a shell whose HOME points inside the run directory."""
    try:
        import pwd
        return Path(pwd.getpwuid(os.getuid()).pw_dir)
    except (ImportError, KeyError):
        return Path.home()


def main(argv=None) -> int:
    """`python3 -m orchard.opinventory`: run the inventory into a stage directory, then record it in delta.json.
    Exit 0 or 2 as run_inventory does. A missing delta.json is not an error here: the inventory files are still written."""
    ap = argparse.ArgumentParser(prog="python3 -m orchard.opinventory", description=main.__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True, help="the stage directory")
    ap.add_argument("--python", required=True, help="interpreter with torch and transformers")
    ap.add_argument("--ttnn-python", default=str(_real_home() / ".tenstorrent-venv" / "bin" / "python"),
                    help="the shared venv's interpreter, asked only for the installed ttnn's version "
                         "(default ~/.tenstorrent-venv/bin/python under the operator's home)")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seq", type=int, default=8)
    ap.add_argument("--class", dest="cls", default=None)
    a = ap.parse_args(argv)
    rc = run_inventory(a.model_dir, python=a.python, ttnn_python=a.ttnn_python, out_dir=a.out,
                       layers=a.layers, seq=a.seq, cls=a.cls)
    if rc == 0 and not record_in_delta(a.out):
        print("delta.json was not updated: it or ops-inventory.json is missing or unreadable")
    return rc


if __name__ == "__main__":
    sys.exit(main())
