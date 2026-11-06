#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Find Hugging Face models that derive from a named parent model, or whose name matches a search.

Typical uses:

    # everything built on Qwen3.8-27B (finetunes, quantizations, adapters, merges), newest first
    python3 scripts/hf_find_models.py --parent Qwen/Qwen3.8-27B

    # the same, only models created in the last 14 days
    python3 scripts/hf_find_models.py --parent Qwen/Qwen3.8-27B --since 14

    # a name search: every word must appear in the repo id
    python3 scripts/hf_find_models.py --search "qwen 27b" --author ccore

    # the built-in watchlist of parents we care about, as JSON, for a script or a cron job
    python3 scripts/hf_find_models.py --default-parents --since 7 --json --fail-on-empty

What it asks the Hub: for each parent P it lists models tagged `base_model:P`, `base_model:finetune:P`, `base_model:quantized:P`,
`base_model:adapter:P` and `base_model:merge:P`. For each --search it runs the Hub's text search (the first word) and keeps the results
whose repo id contains every word. For each --author it lists that author's models. The results are merged, de-duplicated, and sorted
by creation date, newest first. Each result says which query found it.

Limits: the Hub only knows the parent when the repo's card names its `base_model`, so a repo with no card metadata is found only by name.
A repo that is not on the Hub yet cannot be found. The script reads only public listing data. When HF_TOKEN is set it is sent as a
bearer header so gated and private-to-you repos show up; the token is never printed and never put in a URL.

Exit status: 0 normally; 1 with --fail-on-empty when nothing matched; 2 for a usage error.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

API = "https://huggingface.co/api/models"
KINDS = ("", "finetune", "quantized", "adapter", "merge")
# Parents the project watches. Edit this tuple when the project starts caring about another model.
DEFAULT_PARENTS = ("Qwen/Qwen3.8-27B", "Qwen/Qwen3-Coder-Next", "Qwen/Qwen3.5-35B-A3B")
# Tags worth showing in the table: Tenstorrent-related, file format, and the license.
INTERESTING_TAGS = ("blackhole", "wormhole", "tenstorrent", "tt-model-cache", "gguf", "dflash", "thin", "vllm", "p150x4", "p300c")
REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
AUTHOR_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
EXPAND = ("createdAt", "lastModified", "downloads", "tags", "gated", "pipeline_tag")


def http_get(url: str, headers: dict) -> bytes:
    """Fetch one URL and return the body. Replaced by a fake in the tests."""
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 - fixed https host
        return resp.read()


def _url(params: dict, limit: int) -> str:
    items = [(k, v) for k, v in params.items()]
    items += [("expand[]", f) for f in EXPAND]
    items += [("limit", str(limit)), ("sort", "createdAt"), ("direction", "-1")]
    return API + "?" + urllib.parse.urlencode(items)


def _parse_date(text):
    if not isinstance(text, str):
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def find(*, parents, searches, authors, since_days, limit, fetch=http_get, now=None, token=None):
    """Run every query. Returns (models, errors): models newest first, each with a `found_by` list; errors one line per failed query.

    A failing query does not stop the others. A parent that is not `owner/name` raises ValueError before anything is fetched, so it can
    never change the rest of the URL."""
    for p in parents:
        if not REPO_RE.match(p):
            raise ValueError(f"parent must look like owner/name, got {p!r}")
    for a in authors:
        if not AUTHOR_RE.match(a):
            raise ValueError(f"author must be a plain name, got {a!r}")
    now = now or datetime.now(timezone.utc)
    headers = {"Accept": "application/json", "User-Agent": "tt-orchard-hf-find-models"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    queries = []  # (label, url, word filter)
    for p in parents:
        for kind in KINDS:
            tag = f"base_model:{kind}:{p}" if kind else f"base_model:{p}"
            queries.append((f"parent:{p} ({kind or 'any'})", _url({"filter": tag}, limit), None))
    for s in searches:
        words = [w.lower() for w in s.split() if w]
        if words:
            queries.append((f"search:{s}", _url({"search": words[0]}, limit), words))
    for a in authors:
        queries.append((f"author:{a}", _url({"author": a}, limit), None))
    merged: dict = {}
    errors: list = []
    for label, url, words in queries:
        try:
            rows = json.loads(fetch(url, headers))
        except Exception as exc:  # report and go on: one bad query must not hide the rest
            errors.append(f"{label}: {type(exc).__name__}: {str(exc)[:120]}")
            continue
        for row in rows if isinstance(rows, list) else []:
            mid = row.get("id") or row.get("modelId")
            if not isinstance(mid, str):
                continue
            if words and not all(w in mid.lower() for w in words):
                continue
            entry = merged.setdefault(mid, dict(row, id=mid, found_by=[]))
            # The `any` parent query already covers the specific kinds; keep one label per (query kind) found.
            if label not in entry["found_by"]:
                entry["found_by"].append(label)
    models = list(merged.values())
    if since_days is not None:
        cutoff = now - timedelta(days=since_days)
        models = [m for m in models if (_parse_date(m.get("createdAt")) is None or _parse_date(m.get("createdAt")) >= cutoff)]
    models.sort(key=lambda m: _parse_date(m.get("createdAt")) or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    for m in models:
        m["found_by"] = _compact_labels(m["found_by"])
    return models, errors


def _compact_labels(labels):
    """`parent:P (any)` is implied by a more specific `parent:P (finetune)` and so on: drop it when a specific one is present."""
    specific = {l.split(" (")[0] for l in labels if l.startswith("parent:") and not l.endswith("(any)")}
    return [l for l in labels if not (l.endswith("(any)") and l.split(" (")[0] in specific)]


def render_table(models) -> str:
    if not models:
        return "No matching models.\n"
    rows = []
    for m in models:
        d = _parse_date(m.get("createdAt"))
        tags = [t for t in (m.get("tags") or []) if t in INTERESTING_TAGS or t.startswith("license:")]
        if m.get("gated"):
            tags.append("gated")
        rows.append((m["id"], d.strftime("%Y-%m-%d") if d else "?", str(m.get("downloads", "?")), ",".join(tags), "; ".join(m["found_by"])))
    head = ("repo", "created", "downloads", "tags", "found by")
    widths = [max(len(head[i]), *(len(r[i]) for r in rows)) for i in range(5)]
    line = lambda r: "  ".join(r[i].ljust(widths[i]) for i in range(5)).rstrip()  # noqa: E731
    return "\n".join([line(head), line(tuple("-" * w for w in widths)), *(line(r) for r in rows)]) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Find Hugging Face models by parent model or name.")
    ap.add_argument("--parent", action="append", default=[], metavar="OWNER/NAME", help="list models derived from this parent (repeatable)")
    ap.add_argument("--default-parents", action="store_true", help="also use the built-in watchlist of parents")
    ap.add_argument("--search", action="append", default=[], metavar="WORDS", help="name search; every word must be in the repo id (repeatable)")
    ap.add_argument("--author", action="append", default=[], help="list this author's models (repeatable)")
    ap.add_argument("--since", type=float, default=None, metavar="DAYS", help="only models created in the last DAYS days")
    ap.add_argument("--limit", type=int, default=50, help="results per query (default 50)")
    ap.add_argument("--json", action="store_true", help="print one JSON document instead of a table")
    ap.add_argument("--fail-on-empty", action="store_true", help="exit 1 when nothing matched")
    args = ap.parse_args(argv)
    parents = list(args.parent) + (list(DEFAULT_PARENTS) if args.default_parents else [])
    if not (parents or args.search or args.author):
        ap.error("give at least one of --parent, --default-parents, --search, --author")
    try:
        models, errors = find(parents=parents, searches=args.search, authors=args.author, since_days=args.since, limit=args.limit,
                              fetch=http_get, token=os.environ.get("HF_TOKEN") or None)
    except ValueError as exc:
        ap.error(str(exc))
    if args.json:
        print(json.dumps({"models": models, "errors": errors}, indent=2, sort_keys=True))
    else:
        sys.stdout.write(render_table(models))
        for e in errors:
            print(f"warning: {e}", file=sys.stderr)
    return 1 if (args.fail_on_empty and not models) else 0


if __name__ == "__main__":
    sys.exit(main())
