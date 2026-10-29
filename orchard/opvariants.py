# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Find the ttnn names that might implement an aten op the mapping table has no entry for.

The op inventory (orchard/opinventory.py) calls an op `unmapped` when the hand-kept table has no row for it, however
close a ttnn name is. This module reads the dump of ttnn's names and proposes up to three of them for each such op, each
with the reason it was proposed. A proposal is a guess about a name. It is never `mapped`: only a table row makes an
op `mapped`, and `tt-orchard inventory verify-candidates` tests the guess on small tensors before a person adds a row.

How a name is read (all tables are in orchard/data/op_variants.json):

    aten side   `aten.__and__.Tensor` -> base `__and__`, overload `Tensor`. A trailing `_` marks an in-place op. Dunder
                names are read without their underscores (`and`). Names are split into tokens at `_` and at camelCase.
    ttnn side   the namespace (`experimental.`, `transformer.`) and the prefixes `moreh_`, `fused_`, `dit_`, `wan_` are
                dropped, then the name is split into tokens the same way. The key of a name is its tokens joined without
                underscores, so `rms_norm`, `rmsnorm` and `RMSNorm` have one key.

Candidates come in five tiers. The first tier that has any candidate decides the answer, so a close name is never listed
beside an exact one:

    normalised-equal  the keys are equal (an in-place op lists the in-place ttnn name first)
    synonym           the synonym table maps the aten op (or `op.overload`) to a ttnn key that exists
    token-subset      one side's tokens are inside the other's and the extra tokens are only qualifiers
                      (`and` -> `bitwise_and`; `split_with_sizes` -> `split`); `index` does not reach `index_fill`
    near              the keys differ by at most one letter (two for keys of eight letters or more)
    affix             one key is the start or the end of the other and the other is at most six letters longer

Names that carry a token of a different kind of operation (`all` of all_gather, `bw` of a backward op, `device`, `mesh`,
`config` and so on) are left out of the last three tiers.

Standard library only.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from orchard import safeio

MAX_DATA_BYTES = 4 * 1024 * 1024        # choice: the shipped table is a few kilobytes

DATA_FILE = Path(__file__).resolve().parent / "data" / "op_variants.json"
WHY = ("normalised-equal", "synonym", "token-subset", "near", "affix")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


@lru_cache(maxsize=1)
def load_data(path=DATA_FILE) -> dict:
    try:
        return safeio.read_json(safeio.resolved(path), max_bytes=MAX_DATA_BYTES)
    except safeio.SafeReadError as exc:
        raise ValueError(f"{path}: {exc.reason}") from exc


def tokens(text: str) -> list:
    """Lower-case tokens of a name: camelCase and `_` split it, digits stay with their letters."""
    parts = _CAMEL.sub("_", text).lower()
    return [t for t in re.split(r"[^a-z0-9]+", parts) if t]


def parse_aten(op: str) -> dict:
    """`aten.__and__.Tensor` -> {base, overload, inplace, tokens, key}. `base` keeps a leading `_` or dunder so the
    synonym table can name it; `tokens` and `key` do not use them."""
    s = op.strip()
    for pre in ("aten::", "aten."):
        if s.startswith(pre):
            s = s[len(pre):]
            break
    base, _, overload = s.partition(".")
    if overload == "default":
        overload = ""
    dunder = len(base) > 4 and base.startswith("__") and base.endswith("__")
    inplace = base.endswith("_") and not dunder
    core = base[:-1] if inplace else base
    toks = tokens(core)
    return {"base": core, "overload": overload, "inplace": inplace, "tokens": toks, "key": "".join(toks)}


def parse_ttnn(name: str, data=None) -> dict:
    """One dump name -> {name, tokens, key, inplace, blocked, skip}. `skip` is a name that is an API class or helper
    spelled in CamelCase (`CloseDevice`): never proposed."""
    d = data or load_data()
    short = name.rsplit(".", 1)[-1]
    low, legacy = short, False
    for pre in d["ttnn_noise_prefixes"]:
        if low.startswith(pre):
            low, legacy = low[len(pre):], True
            break
    inplace = low.endswith("_")
    toks = tokens(low[:-1] if inplace else low)
    blocked = any(t in set(d["ttnn_blocked_tokens"]) for t in toks)
    return {"name": name, "tokens": toks, "key": "".join(toks), "inplace": inplace, "blocked": blocked,
            "namespaced": "." in name, "legacy": legacy, "skip": short[:1].isupper()}


def distance(a: str, b: str) -> int:
    """Levenshtein distance."""
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class Index:
    """The dump's names, parsed once."""

    def __init__(self, names, data=None):
        self.data = data or load_data()
        self.entries = [e for e in (parse_ttnn(n, self.data) for n in sorted(set(names))) if not e["skip"] and e["key"]]

    def by_key(self, key):
        return [e for e in self.entries if e["key"] == key]


def _order(entries, inplace):
    """The spellings of one key worth proposing, best first. Plain names come before namespaced ones and before the
    legacy ones (`moreh_`), and a spelling is dropped when a better one has the same in-place form. An in-place aten op
    keeps the best in-place spelling and the best plain one; any other op keeps no in-place spelling while a plain one
    exists."""
    ranked = sorted(entries, key=lambda e: (e["inplace"] != inplace, e["legacy"], e["namespaced"], len(e["name"]),
                                            e["name"]))
    kept, seen = [], set()
    for e in ranked:
        if e["inplace"] not in seen:
            seen.add(e["inplace"])
            kept.append(e)
    if not inplace and any(not e["inplace"] for e in kept):
        kept = [e for e in kept if not e["inplace"]]
    return kept


def _synonym_keys(a, data):
    table = data["synonyms"]
    hits = []
    for k in ((a["base"] + "." + a["overload"]) if a["overload"] else None, a["base"], a["base"].strip("_")):
        if k and k in table:
            hits = table[k]
            break
    return ["".join(tokens(h)) for h in hits]


def candidates(op: str, index: Index) -> list:
    """Up to `max_candidates` ttnn names for one aten op, each {"ttnn", "why"}; [] when nothing is close.
    `why` is one of WHY. See the module text for the tiers."""
    d = index.data
    a = parse_aten(op)
    if not a["key"]:
        return []
    limit = d["max_candidates"]
    out = []

    def done(entries, why):
        for e in _order(entries, a["inplace"])[:limit]:
            out.append({"ttnn": e["name"], "why": why})
        return bool(out)

    # tiers 1 and 2 together: the exact names, then what the synonym table says
    for e in _order(index.by_key(a["key"]), a["inplace"]):
        out.append({"ttnn": e["name"], "why": "normalised-equal"})
    per_target = [_order(index.by_key(k), a["inplace"]) for k in _synonym_keys(a, d)]
    # the best entry of each synonym target first, so one target's spellings do not crowd out the others
    for rank in range(max([len(t) for t in per_target] or [0])):
        for t in per_target:
            if rank < len(t) and all(c["ttnn"] != t[rank]["name"] for c in out):
                out.append({"ttnn": t[rank]["name"], "why": "synonym"})
    if out:
        return out[:limit]
    live = [e for e in index.entries if not e["blocked"]]
    # tier 3
    atok = set(a["tokens"])
    tq, aq = set(d["ttnn_qualifier_tokens"]), set(d["aten_qualifier_tokens"])
    sub = []
    for e in live:
        etok = set(e["tokens"])
        if atok < etok and (etok - atok) <= tq:
            sub.append(e)
        elif etok < atok and (atok - etok) <= aq and len(e["key"]) >= 3:
            sub.append(e)
    if done(sub, "token-subset"):
        return out
    # tier 4
    k = a["key"]
    if len(k) >= d["near_min_length"]:
        bound = d["near_max_distance_long"] if len(k) >= d["near_long_from_length"] else d["near_max_distance_short"]
        near = [e for e in live if len(e["key"]) >= d["near_min_length"] and distance(k, e["key"]) <= bound]
        if done(near, "near"):
            return out
    # tier 5
    m = d["affix_min_length"]
    aff = []
    for e in live:
        s, l = sorted((k, e["key"]), key=len)
        if len(s) >= m and len(l) - len(s) <= d["affix_max_extra"] and s != l and (l.startswith(s) or l.endswith(s)):
            aff.append(e)
    done(aff, "affix")
    return out
