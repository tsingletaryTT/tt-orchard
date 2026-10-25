#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Build the data files the site shows, from real run directories and the committed evidence.

    python3 scripts/make_site_data.py --run ~/orchard-runs/proprietarylegal--qwen3.8-27b-cjk-trim --out site/assets/data

Writes replay.json (the run's narration, line by line, with seconds since the start), stages.json (each stage's wall time), catalog.json (what stage 0 says
about 29 catalog models), and bench.json (the bytkim benchmark). Every line passes a scrub: a home path, the host name and the user name are replaced, and the script
stops if one is left.
"""
import argparse
import datetime
import io
import json
import re
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from orchard import narrate, scrub, ui  # noqa: E402

ANSI = re.compile(r"\x1b\[[0-9;]*m")
LINE = re.compile(r"^(\d\d):(\d\d):(\d\d)\s+(orchardist|grafter|head grower|seasonal hand|sheepdog)\s+(.*)$")


SECRETS = (                        # beyond orchard.scrub's token patterns: an assignment or header that carries a secret, and common key shapes
    (re.compile(r"(?i)\bBearer\s+\S+"), "Bearer <redacted>"),
    (re.compile(r"(?i)\b(token|secret|password|passwd|api[_-]?key|authorization)(\s*[:=]\s*)\S+"), r"\1\2<redacted>"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"), "<redacted>"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "<redacted>"),
)


def scrub_text(text: str, run: Path) -> str:
    host, user = socket.gethostname(), Path.home().name
    for _, pattern in scrub.TOKEN_PATTERNS:
        text = pattern.sub("<redacted>", text)
    for pattern, repl in SECRETS:
        text = pattern.sub(repl, text)
    text = text.replace(str(run), "<run>").replace(str(Path.home()), "~")
    text = re.sub(r"/home/[^\s]*", "~", text)
    text = re.sub(re.escape(host), "this-box", text, flags=re.I)
    return re.sub(rf"\b{re.escape(user)}\b", "user", text)


def assemble(raw_lines: list[str], run: Path) -> list[dict]:
    """Narration lines to rows: ANSI removed, wrapped continuation lines joined to their line, seconds since the first, and the whole message scrubbed."""
    rows: list[dict] = []
    t0, prev, days = None, None, 0
    for raw in raw_lines:
        line = ANSI.sub("", raw)
        m = LINE.match(line)
        if not m:
            if rows and line.startswith(" " * 20):
                rows[-1]["text"] += " " + line.strip()
            continue
        h, mi, s, actor, text = int(m[1]), int(m[2]), int(m[3]), m[4], m[5]
        sec = h * 3600 + mi * 60 + s
        if t0 is None:
            t0 = sec
        elif sec < prev:               # the clock wrapped past midnight (consecutive lines are less than a day apart)
            days += 1
        prev = sec
        rows.append({"t": days * 86400 + sec - t0, "actor": actor, "text": text.strip()})
    for r in rows:                      # scrub the assembled message: a wrapped line may carry a name that the first line does not
        r["text"] = scrub_text(r["text"], run)
    return rows


def replay(run: Path) -> list[dict]:
    out = io.StringIO()
    narrate.Narrator(run, ui.Style(color=False, emoji=False), out, replay=True, tz=datetime.timezone.utc).poll()
    return assemble(out.getvalue().splitlines(), run)


def stages(run: Path) -> list[dict]:
    names = {}
    starts, ends = {}, {}
    for line in (run / "ledger.jsonl").read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        n = e.get("stage")
        if n is None:
            continue
        ts = datetime.datetime.strptime(e["ts"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc).timestamp()
        if e["event"] == "stage_start" and n not in starts:
            starts[n] = ts
        if e["event"] == "stage_end":
            ends[n] = (ts, e["data"].get("result"))
    from orchard.stages import STAGES
    from orchard import lexicon
    rows = []
    for spec in STAGES:
        n = spec.number
        if n in ends and n in starts:
            rows.append({"n": n, "name": spec.name, "orchard": lexicon.STAGE_NAMES.get(n) if hasattr(lexicon, "STAGE_NAMES") else None,
                         "seconds": round(ends[n][0] - starts[n]), "result": ends[n][1]})
        elif n in ends:
            rows.append({"n": n, "name": spec.name, "seconds": 0, "result": ends[n][1]})
    return rows


def cut_at_ready(rows: list[dict]) -> list[dict]:
    """The replay ends where the run did: hardware released after the operator bundle. Packaging commands that came hours later are another story."""
    for i, r in enumerate(rows):
        if "ready for operator review" in r["text"]:
            tail = [j for j in range(i, len(rows)) if "hardware released" in rows[j]["text"]]
            return rows[: (tail[0] if tail else i) + 1]
    return rows


def bench() -> dict:
    d = json.loads((ROOT / "docs/run-logs/evidence/2026-10-21-bytkim-pi-benchmark-rerun/benchmark.json").read_text(encoding="utf-8"))
    out = {"models": {"pi": "bytkim/Qwen3.8-27B-pi", "base": "Qwen/Qwen3.8-27B"}, "tasks": {}}
    for name, label in (("humaneval", "HumanEval"), ("mbpp", "MBPP"), ("ifeval", "IFEval"), ("gsm8k", "GSM8K"), ("bfcl_simple", "BFCL simple"), ("bfcl_multiple", "BFCL multiple")):
        t = d["tasks"][name]
        out["tasks"][name] = {"label": label, "n": t["pi"]["n"], "pi": t["pi"]["score"], "base": t["base"]["score"], "se_pi": t["pi"]["stderr"],
                              "se_base": t["base"]["stderr"], "p": (t.get("paired") or {}).get("sign_test_p"),
                              "cut_pi": t["pi"].get("cut_off"), "cut_base": t["base"].get("cut_off")}
    return out


def catalog() -> list[dict]:
    path = Path(__file__).resolve().parent / "data" / "catalog-replay.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rp = cut_at_ready(replay(a.run))
    (a.out / "replay.json").write_text(json.dumps({"run": a.run.name.replace("proprietarylegal--", ""), "lines": rp}, indent=1), encoding="utf-8")
    (a.out / "stages.json").write_text(json.dumps(stages(a.run), indent=1), encoding="utf-8")
    (a.out / "catalog.json").write_text(json.dumps(catalog(), indent=1), encoding="utf-8")
    (a.out / "bench.json").write_text(json.dumps(bench(), indent=1), encoding="utf-8")
    for f in a.out.glob("*.json"):
        text = f.read_text(encoding="utf-8")
        assert "/home/" not in text and socket.gethostname().lower() not in text.lower(), f"{f.name} still holds a path or the host name"
        assert not re.search(rf"\b{re.escape(Path.home().name)}\b", text), f"{f.name} still holds the user name"
        hits = scrub.scrub_text(text, hostname=socket.gethostname(), home=str(Path.home()))
        assert not hits, f"{f.name} still holds {', '.join(hits)}"
    print(f"replay {len(rp)} lines, {rp[-1]['t'] if rp else 0} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
