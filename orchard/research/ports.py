# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Has anyone already ported this architecture?

Sources: the tt-model community catalog (`tt model search --catalog`), tt-metal's `models/` listings and
models README, GitHub issues and pull requests for the architecture name, GitHub code search through `gh`
(code search needs a login; the other calls do not), and the code a published bundle ships under
`code/models/`. A search term comes from the new model's own config, so it is checked against SAFE_TERM
before it reaches a URL or a command line."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from urllib.parse import quote

from orchard import nearest
from orchard.research.http import FetchError
from orchard.research.types import INFERRED, WEAK, Evidence, Result, Unavailable, clean, valid_id

SAFE_TERM = re.compile(r"[A-Za-z0-9_.-]{3,80}")
METAL = "tenstorrent/tt-metal"
CONTENTS = "https://api.github.com/repos/" + METAL + "/contents/models/{sub}"
README_URL = "https://raw.githubusercontent.com/" + METAL + "/main/models/README.md"
ISSUES_URL = "https://api.github.com/search/issues?q={q}&per_page={n}"
GITHUB_PREFIX = "https://github.com/"
README_ROW = re.compile(r"^\|\s*\[([^\]]+)\]\((https://[^)\s]+)\)")
NOT_A_MODEL = frozenset({"common", "utils", "perf"})      # shared code in a bundle that is not a model port
GH_TIMEOUT_S = 30
MAX_ITEMS = 5


def terms(config) -> dict:
    """Search terms from a model's config: `specific` (gemma4), `family` (gemma) and `classes` (the exact
    architecture class names). Anything that does not match SAFE_TERM is left out."""
    config = config if isinstance(config, dict) else {}
    model_type = config.get("model_type") if isinstance(config.get("model_type"), str) else ""
    first = model_type.lower().split("_")[0]
    letters = re.match(r"[a-z]+", first)
    specific = [first] if SAFE_TERM.fullmatch(first) and letters and first != letters.group(0) else []
    family = [letters.group(0)] if letters and SAFE_TERM.fullmatch(letters.group(0)) else []
    arch = config.get("architectures")
    classes = [a for a in arch if isinstance(a, str) and SAFE_TERM.fullmatch(a)] if isinstance(arch, list) else []
    return {"specific": specific, "family": family, "classes": classes[:2]}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def match_names(names, t: dict) -> list[tuple[str, str]]:
    """[(name, confidence)]: inferred when a name holds the specific term (gemma4), weak when it holds only
    the family (gemma)."""
    out = []
    for name in names:
        norm = _norm(str(name))
        if any(s in norm for s in t["specific"]):
            out.append((name, INFERRED))
        elif any(f in norm for f in t["family"]):
            out.append((name, WEAK))
    return sorted(out, key=lambda p: p[1] != INFERRED)


def readme_rows(text: str) -> list[tuple[str, str]]:
    """[(model name, link)] from the models README's performance table."""
    rows = []
    for line in text.splitlines():
        m = README_ROW.match(line)
        if m:
            rows.append((re.sub(r"\s*\(.*?\)\s*$", "", m.group(1)).strip(), m.group(2)))
    return rows


def metal(fetch, t: dict) -> Result:
    res = Result()
    for sub in ("demos", "experimental"):
        url = CONTENTS.format(sub=sub)
        try:
            items = fetch.json(url)
        except FetchError as exc:
            res.unavailable.append(Unavailable("tt-metal", f"models/{sub}: {exc.reason}"))
            continue
        if not isinstance(items, list):
            res.unavailable.append(Unavailable("tt-metal", f"models/{sub}: the answer is not a listing"))
            continue
        by_name = {i["name"]: i for i in items if isinstance(i, dict) and isinstance(i.get("name"), str)}
        for name, conf in match_names(list(by_name), t):
            link = by_name[name].get("html_url")
            res.evidence.append(Evidence(
                "tt-metal", clean(f"tt-metal has models/{sub}/{name}"),
                link if isinstance(link, str) and link.startswith(GITHUB_PREFIX) else GITHUB_PREFIX + METAL,
                conf, fetch.now()))
    try:
        text = fetch.get(README_URL, max_bytes=2 * 1024 * 1024).decode("utf-8", "replace")
    except FetchError as exc:
        res.unavailable.append(Unavailable("tt-metal", f"models/README.md: {exc.reason}"))
        return res
    rows = readme_rows(text)
    by_name = {n: u for n, u in rows}
    for name, conf in match_names(list(by_name), t):
        res.evidence.append(Evidence("tt-metal", clean(f"tt-metal's models README lists {name}"),
                                     by_name[name] if by_name[name].startswith(GITHUB_PREFIX) else README_URL,
                                     conf, fetch.now()))
    return res


def issues(fetch, t: dict, *, limit: int = MAX_ITEMS) -> Result:
    res = Result()
    for term in (t["specific"] or t["family"])[:1] + t["classes"][:1]:
        url = ISSUES_URL.format(q=quote(f"{term} repo:{METAL}", safe="+:"), n=limit)
        try:
            data = fetch.json(url)
        except FetchError as exc:
            res.unavailable.append(Unavailable("github issues", f"{term}: {exc.reason}"))
            continue
        for item in (data.get("items") if isinstance(data, dict) else None) or []:
            if not isinstance(item, dict) or not str(item.get("html_url", "")).startswith(GITHUB_PREFIX):
                continue
            kind = "PR" if "pull_request" in item else "issue"
            res.evidence.append(Evidence(
                "github", clean(f"{kind} #{item.get('number')} ({item.get('state')}): {item.get('title')}"),
                item["html_url"], WEAK, fetch.now()))
    return res


def github_code(t: dict, *, run=subprocess.run, which=shutil.which, now: str = "") -> Result:
    """Code search needs a login, so it goes through `gh` (read only). Without gh, or when gh fails, it is
    reported as unavailable."""
    res = Result()
    exe = which("gh")
    if not exe:
        res.unavailable.append(Unavailable("github code search", "gh is not installed"))
        return res
    for term in t["classes"][:1]:
        argv = [exe, "api", "-X", "GET", "search/code", "-f", f"q={term} repo:{METAL}", "-f", f"per_page={MAX_ITEMS}"]
        try:
            done = run(argv, capture_output=True, text=True, timeout=GH_TIMEOUT_S)
        except (OSError, subprocess.SubprocessError) as exc:
            res.unavailable.append(Unavailable("github code search", f"gh failed: {type(exc).__name__}"))
            continue
        if done.returncode != 0:
            first = (done.stderr or "").strip().splitlines()[:1]
            res.unavailable.append(Unavailable("github code search", clean("gh failed: " + (first[0] if first else f"exit {done.returncode}"))))
            continue
        try:
            items = json.loads(done.stdout).get("items") or []
            if not isinstance(items, list):
                raise ValueError("items is not a list")
        except (ValueError, AttributeError):
            res.unavailable.append(Unavailable("github code search", "gh gave no JSON"))
            continue
        for item in items[:MAX_ITEMS]:
            if isinstance(item, dict) and str(item.get("html_url", "")).startswith(GITHUB_PREFIX):
                res.evidence.append(Evidence("github", clean(f"tt-metal code mentions {term} in {item.get('path')}"),
                                             item["html_url"], INFERRED, now))
    return res


def catalog(family: str, *, run=subprocess.run, which=shutil.which, now: str = "") -> Result:
    """Bundles in the community catalog for this family: `tt model search <family> --catalog --json`, or, when `tt`
    is missing or gives no answer, `tt-model search <family> --catalog --json` (the order `nearest.search` uses).
    None of the boxes this was run on has `tt`, so without the fallback the catalog row was always unavailable."""
    res = Result(data={"bundles": []})
    # The family becomes an argv element of `tt` or `tt-model`. A leading dash would be read as an option.
    if not (isinstance(family, str) and SAFE_TERM.fullmatch(family) and not family.startswith("-")):
        res.unavailable.append(Unavailable("catalog", "the search term is not a plain word, so no search was made"))
        return res
    attempts = [(label, [exe, *sub, family, "--catalog", "--json"])
                for label, name, sub in (("tt model search", "tt", ["model", "search"]),
                                         ("tt-model search", "tt-model", ["search"]))
                if (exe := which(name))]
    if not attempts:
        res.unavailable.append(Unavailable("catalog", "neither `tt` nor `tt-model` is installed"))
        return res
    got, failures = None, []
    for label, argv in attempts:
        try:
            done = run(argv, capture_output=True, text=True, timeout=nearest.SEARCH_TIMEOUT_S)
        except (OSError, subprocess.SubprocessError) as exc:
            failures.append(f"`{label} --catalog` failed: {type(exc).__name__}")
            continue
        got = nearest.parse_search(done.stdout) if done.returncode == 0 else None
        if got is not None:
            break
        failures.append(f"`{label} --catalog` gave no answer (exit {done.returncode})")
    if got is None:
        res.unavailable.append(Unavailable("catalog", "; ".join(failures)))
        return res
    skipped = 0
    for b in got:
        # The search output is untrusted; a name that is not a Hub id never becomes a URL or a bundle row.
        if not valid_id(b.get("name")):
            skipped += 1
            continue
        res.data["bundles"].append(b)
        res.evidence.append(Evidence("catalog", clean(f"the community catalog lists {b['name']}"),
                                     f"https://huggingface.co/{b['name']}", INFERRED, now))
    if skipped:
        res.unavailable.append(Unavailable("catalog", clean(f"{skipped} catalog name(s) skipped: not a Hub id")))
    return res


def bundle_code(siblings, bundle: str, now: str) -> list[Evidence]:
    """A bundle's file list names the code it ships: code/models/autoports/<model>/ or code/models/demos/<model>/."""
    if not valid_id(bundle):                                  # the id goes into a URL
        return []
    seen: list[str] = []
    for item in siblings if isinstance(siblings, list) else []:
        name = item.get("rfilename") if isinstance(item, dict) else None
        m = re.match(r"code/models/(?:autoports|demos)/([A-Za-z0-9_.-]+)/", name or "")
        if m and m.group(1) not in seen and m.group(1) not in NOT_A_MODEL:
            seen.append(m.group(1))
    return [Evidence("bundle", clean(f"bundle {bundle} ships code for {model}"), f"https://huggingface.co/{bundle}",
                     INFERRED, now) for model in seen]


def collect(fetch, config, *, family: str, run=subprocess.run, which=shutil.which, no_github: bool = False) -> Result:
    t = terms(config)
    res = Result()
    res.extend(metal(fetch, t))
    cat = catalog(family, run=run, which=which, now=fetch.now())
    res.extend(cat)
    res.data["catalog"] = cat.data["bundles"]
    if not no_github:
        res.extend(issues(fetch, t))
        res.extend(github_code(t, run=run, which=which, now=fetch.now()))
    return res
