# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`tt-orchard reported show|accept RUN`: the scores a model's authors report, from the model card to a package card (issue 19).

Stage 0 drafts the rows (`stages/0/evidence/reported-results.draft.json`, written by the delta-triage template). The draft
is the authors' claims, quoted with their lines, and nobody has confirmed it. `accept` copies the rows a person chooses into
`<run>/reported-results.json`, the one file `package` and `package-card` read, and records in the ledger who accepted what.
Neither `package` nor `package-card` accepts anything by itself.

The draft sits in a directory a stage 0 agent can write, so `accept` does not trust it. It reads the model card again and
runs the same extractor over it: a row is accepted only when the card on disk gives exactly that name, value and line. When
the card is not on disk the rows are accepted and the ledger says they were not checked. The rows then pass the rules the
package card applies to reported-results.json (orchard/package.py, `read_reported_file`). The command reads files in the run,
opens no network connection and starts nothing.

Exit codes: 0 done, 2 refused (nothing was written).
"""
from __future__ import annotations

import getpass
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

from orchard.ledger import Ledger, LedgerLocked
from orchard.package import REPORTED_FILE, PackageError, read_reported_file
from orchard.package_cmd import PackageCommandError, _entries, _replace_file

DRAFT = "stages/0/evidence/reported-results.draft.json"
TEMPLATE = Path(__file__).resolve().parent / "skills" / "delta-triage-templates" / "delta_triage.py"


class Refused(Exception):
    """The command refuses; the message says why."""


def _template():
    """The shipped delta-triage template, imported by path. It holds the extractor and the plain-number rule, so the rule
    that wrote the draft is the rule that checks it."""
    spec = importlib.util.spec_from_file_location("delta_triage_for_reported_cmd", TEMPLATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _say(out, text: str = "") -> None:
    print(text, file=out, flush=True)


def _plural(n: int, one: str, many: str) -> str:
    return one if n == 1 else many


def _load_draft(run: Path) -> tuple[dict, Path, bytes]:
    path = run / DRAFT
    if not path.is_file():
        raise Refused(f"{path} does not exist: stage 0 wrote no draft of the reported scores for this run (the model card "
                      "had no table or model-index of scores for the model)")
    raw = path.read_bytes()
    try:
        draft = json.loads(raw)
    except ValueError as exc:
        raise Refused(f"{path} is not readable JSON: {exc}") from None
    rows = draft.get("rows") if isinstance(draft, dict) else None
    ok = isinstance(rows, list) and isinstance(draft.get("source"), str) and isinstance(draft.get("model"), str) and all(
        isinstance(r, dict) and isinstance(r.get("name"), str) and isinstance(r.get("value"), str)
        and isinstance(r.get("note", ""), str) for r in rows)
    if not ok:
        raise Refused(f"{path} is not a draft of reported scores (it needs a model, a source and rows with a name and a "
                      "text value)")
    return draft, path, raw


def _card_text(run: Path) -> str | None:
    """The model card stage 0 read, found through the snapshot stage 0's configuration names. None when it is not on disk."""
    try:
        config = json.loads((run / "stages" / "0" / "triage_config.json").read_text(encoding="utf-8"))
        readme = Path(config["model_snapshot"]) / "README.md"
        with open(readme, "rb") as fh:
            return fh.read(1 << 21).decode("utf-8", errors="replace")
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _recorded_model(entries: list[dict]) -> str:
    """The model the run was started for, from its ledger. The draft's own `model` field is an agent-writable claim."""
    for e in entries:
        if e["event"] == "run_start" and isinstance(e["data"].get("model"), str) and e["data"]["model"].strip():
            return e["data"]["model"].strip()
    raise Refused("the ledger records no model (no run_start with a model), so the draft cannot be checked against the card")


def _repo(model: str) -> str:
    return model.split("@")[0].strip().lower()


def _check_rows(chosen: list[dict], draft: dict, run: Path, model: str) -> bool:
    """Refuse a row that is not a plain number, or that the model card on disk does not give. True when every row was
    checked against the card, False when the card is not on disk."""
    tpl = _template()
    problems = []
    for row in chosen:
        value = row["value"]
        if not tpl._plain_number(value) or len(value) > tpl.MAX_REPORTED_VALUE:
            problems.append(f"{row['name']!r}: {value!r} is not a plain number")
    if problems:
        raise Refused("; ".join(problems))
    text = _card_text(run)
    if text is None:
        return False
    found = {(r["name"], r["value"], r["extracted_from"]["line"]) for r in tpl.extract_reported(text, model)["rows"]}
    for row in chosen:
        line = (row.get("extracted_from") or {}).get("line")
        if (row["name"], row["value"], line) not in found:
            problems.append(f"{row['name']!r} = {row['value']!r} (line {line})")
    if problems:
        raise Refused("the model card on disk does not give " + "; ".join(problems) + ". The draft was changed, or the "
                      "card was. Run stage 0 again to draft the rows from the card as it is")
    return True


def _choose(draft: dict, args) -> list[dict]:
    rows = draft["rows"]
    if not rows:
        raise Refused("the draft has no rows: the card gave no score for this model that could be read")
    if getattr(args, "all", False):
        return list(rows)
    wanted = [w.strip() for w in (args.rows or "").split(",") if w.strip()] + list(args.row or [])
    if not wanted:
        raise Refused("say which rows: --all, --rows NAME,NAME or --row NAME")
    names = [r["name"] for r in rows]
    unknown = [w for w in wanted if w not in names]
    if unknown:
        raise Refused(f"the draft has no row named {', '.join(map(repr, unknown))}. The rows are: "
                      + "; ".join(map(repr, names)))
    return [r for r in rows if r["name"] in set(wanted)]


def _file_text(source: str, chosen: list[dict]) -> str:
    return json.dumps({"source": source, "rows": [{"name": r["name"], "value": r["value"], "note": r.get("note", "")}
                                                  for r in chosen]}, indent=2, ensure_ascii=False) + "\n"


def accept(args, out) -> int:
    run = Path(args.run_dir).resolve()
    model = _recorded_model(_entries(run))                         # refuses a directory with no readable ledger
    draft, draft_path, raw = _load_draft(run)
    if _repo(draft["model"]) != _repo(model):
        raise Refused(f"the draft names {draft['model']}, and this run is for {model}. Scores of another model are not "
                      "accepted")
    chosen = _choose(draft, args)
    verified = _check_rows(chosen, draft, run, model)
    text = _file_text(draft["source"], chosen)
    installed = run / REPORTED_FILE
    with tempfile.TemporaryDirectory(dir=run) as tmp:              # judged by the card's own rules before it is installed
        probe = Path(tmp) / REPORTED_FILE
        probe.write_text(text, encoding="utf-8")
        try:
            read_reported_file(probe)
        except PackageError as exc:
            raise Refused(f"the rows are refused by the card's rules: {exc}") from None
    try:
        led = Ledger(run / "ledger.jsonl")
    except LedgerLocked:
        raise Refused(f"a supervisor holds {run / 'ledger.jsonl'}: the run is still going. Stop it first, or wait for it "
                      "to finish") from None
    with led:
        # Looked at under the ledger lock, so the file and the ledger entry describe the same selection even when two
        # accepts run at once.
        replaced = installed.exists() and installed.read_text(encoding="utf-8") != text
        if replaced and not args.replace:
            raise Refused(f"{installed} already holds other reported scores. Pass --replace to replace them")
        if not installed.exists() or replaced:
            _replace_file(installed, text)
        led.append("decision", 0, decision="reported scores accepted", by=_who(), names=[r["name"] for r in chosen],
                   count=len(chosen), of=len(draft["rows"]), source=draft["source"],
                   draft_sha256=hashlib.sha256(raw).hexdigest(), verified_against_card=verified, replaced=replaced)
    _say(out, f"accepted {len(chosen)} of {len(draft['rows'])} reported {_plural(len(draft['rows']), 'score', 'scores')} into "
              f"{installed.name}, source: {draft['source']}")
    for r in chosen:
        _say(out, f"  {r['name']} = {r['value']}   ({r.get('note', '')})")
    if not verified:
        _say(out, "the model card is not on disk, so these rows were not checked against it. The ledger says so.")
    _say(out, "Nothing was published. The cards show these rows, labelled as the authors' scores and not measured here, the "
              "next time `tt-orchard package` or `tt-orchard package-card` writes them.")
    return 0


def _who() -> str:
    try:
        return getpass.getuser()
    except Exception:                                               # no passwd entry, no USER: say so
        return "unknown"


def show(args, out) -> int:
    run = Path(args.run_dir).resolve()
    _entries(run)
    draft, _, _ = _load_draft(run)
    accepted = set()
    try:
        data = json.loads((run / REPORTED_FILE).read_text(encoding="utf-8"))
        accepted = {r.get("name") for r in data.get("rows", []) if isinstance(r, dict)}
    except (OSError, ValueError, AttributeError):
        pass
    rows = draft["rows"]
    _say(out, f"{draft['source']}: {len(rows)} {_plural(len(rows), 'row', 'rows')} drafted by stage 0, not confirmed by anyone.")
    for r in rows:
        ef = r.get("extracted_from") or {}
        mark = "  [accepted]" if r["name"] in accepted else ""
        _say(out, f"  {r['name']} = {r['value']}{mark}")
        _say(out, f"      {ef.get('kind', '?')} line {ef.get('line', '?')}: {ef.get('text', '')}")
    left = list(draft.get("skipped") or [])
    n = len(left) + int(draft.get("skipped_dropped") or 0)
    if n:
        _say(out, f"{n} {_plural(n, 'entry was', 'entries were')} left out:")
        for s in left[:20]:
            _say(out, f"  line {s.get('line', '?')}: {s.get('reason', '')}")
        if n > 20:
            _say(out, f"  ... and {n - 20} more, all listed in {DRAFT}")
    if draft.get("other_model_rows"):
        _say(out, f"{draft['other_model_rows']} comparison-table {_plural(draft['other_model_rows'], 'row was', 'rows were')} "
                  "about other models and ignored.")
    _say(out, f"To use rows: tt-orchard reported accept {run} --all   (or --rows NAME,NAME)")
    return 0


def main(args, *, out=None) -> int:
    out = out or sys.stdout
    try:
        return show(args, out) if args.action == "show" else accept(args, out)
    except (Refused, PackageCommandError) as exc:
        _say(out, f"refused: {exc}")
        return 2
