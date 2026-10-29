# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`tt-orchard updates`: is each tool and model that orchard runs on the newest one upstream has? Read only.

Operator request (2026-10-29): orchard should know when it runs on old tools, when a new coder model appears, and when a
synced upgrade of tt-lang, tt-metal, tt-model-manager and tt-cli is ready to farm with. This module reports. It installs,
upgrades, pulls and restarts nothing and opens no chip. Each upgrade is printed as text for a person to run.

Layers, outermost first:

- `Tools`: the only way this module reaches the network or starts a process. HTTP is a GET to four allowed hosts. A process is
  `<tool> --version` for three named tools, or a read-only git command. Anything else raises NotReadOnly before it runs. Each call
  has a timeout, all calls share a budget, and offline mode makes no call. Tests hand `Tools` fake `get` and `run` functions.
- `check_*`: one function per tracked item. Each returns a Row and never raises; a failed call gives `unknown (why)`.
- `RULES`: the table of items that must move together. A behind row names the other members and cites the rule.
- `verifications.json` (orchard/verifications.py): a row is `verified` only when a recorded run matches the exact versions.

Everything read from a remote answer is cut to a size, type-checked and stripped of control characters (`clean`) before it
reaches a row. The Hugging Face token is read from the environment, sent only to huggingface.co and never stored or printed.
"""
from __future__ import annotations

import copy
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import quote, urlsplit

from orchard import safeio, skillsfresh, ttlang_pin, verifications

CURRENT, BEHIND, CANDIDATE, UNKNOWN = "current", "behind", "newer-candidate", "unknown"
CALL_TIMEOUT_S = 12.0           # choice: one API call; a healthy one takes under a second
BUDGET_S = 60.0                 # choice: all network calls of one run
BODY_MAX = 2 * 1024 * 1024      # choice: the largest list answer is a few hundred KB
CACHE_HOURS = 6.0               # choice: operator request
LIST_MAX = 300                  # most entries read from one list answer
TEXT_MAX = 160                  # longest remote string kept
HOSTS = ("huggingface.co", "api.github.com", "pypi.org", "pypi.eng.aws.tenstorrent.com")
VERSION_TOOLS = ("tt-model", "tt", "gozer")     # tt is the official CLI (tenstorrent/tt-cli); orchard only asks its version
EXTRA_GIT = frozenset({"describe", "log", "show"})
DEV_INDEX = "https://pypi.eng.aws.tenstorrent.com/tt-lang/"
FIND_COMPATIBLE = "scripts/find-compatible-ttlang.sh in the tenstorrent/tt-lang repository"

# Things that must move together, kept as data: `members` are row keys. Rows name the other members of every rule they
# are in when they are behind or a candidate exists.
RULES = (
    {"id": "R1", "members": ("tt-lang", "tt-metal"),
     "text": "A tt-lang commit builds against one tt-metal ref. A new tt-metal tag needs a tt-lang commit that builds on it "
             f"(find it with {FIND_COMPATIBLE}); a new tt-lang commit needs the tt-metal ref it names. "
             "The verified pair is recorded in orchard/ttlang_pin.py."},
    {"id": "R2", "members": ("tt-metal", "tt-model-manager", "coder"),
     "text": "tt-model-manager bundles carry a tt-metal build. The host ttnn and the bundle must agree on the tt-metal "
             "release before a bundle is served next to host-side tests."},
    {"id": "R3", "members": ("tt-model-manager", "tt-cli"),
     "text": "tt-cli drives bundles that tt-model-manager installs; upgrade them in one step and rerun `tt doctor`."},
)

DEFAULTS = {
    "family": ["qwen3-coder", "qwen3.5-coder", "coder-next"],
    "coder_packages": [],                       # empty: the packages in orchard.setup_machine.CODERS that hold "coder"
    "authors": [],                              # empty: the authors of the coder packages
    "repos": {"tt-lang": "tenstorrent/tt-lang", "tt-metal": "tenstorrent/tt-metal",
              "tt-model-manager": "tenstorrent/tt-model-manager", "tt-cli": "tenstorrent/tt-cli",
              "tt-gozer": "tsingletaryTT/tt-gozer"},
    "cache_hours": CACHE_HOURS,
}


class NotReadOnly(Exception):
    """A request that could change something reached the runner. It is never made."""


class ConfigError(Exception):
    pass


def clean(x, n: int = TEXT_MAX) -> str:
    """A string from a remote answer, safe to print: not a string gives "", control and non-printable characters become
    "?", and the length is cut at `n`."""
    if not isinstance(x, str):
        return ""
    return "".join(c if c.isprintable() else "?" for c in x[:n])


def load_config(path=None) -> dict:
    """DEFAULTS, overridden by a TOML file (default config/updates.toml beside the checkout, when it exists). Unknown keys fail."""
    cfg = copy.deepcopy(DEFAULTS)          # a copy of a constant; this module reads no JSON file, so it is not a reader for the hostile-input registry
    p = Path(path) if path else Path(__file__).resolve().parent.parent / "config" / "updates.toml"
    if not p.exists():
        if path:
            raise ConfigError(f"{p} does not exist")
        return cfg
    import tomllib
    try:
        raw = tomllib.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConfigError(f"{p.name}: {type(exc).__name__}") from None
    for k, v in raw.items():
        if k not in DEFAULTS:
            raise ConfigError(f"{p.name}: unknown key {k!r}; known: {', '.join(DEFAULTS)}")
        if isinstance(DEFAULTS[k], list) and not (isinstance(v, list) and all(isinstance(i, str) for i in v)):
            raise ConfigError(f"{p.name}: {k} must be a list of strings")
        if k == "repos":
            if not isinstance(v, dict) or not all(isinstance(a, str) and isinstance(b, str) for a, b in v.items()):
                raise ConfigError(f"{p.name}: repos must map names to owner/name strings")
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg


# ---- the runner -------------------------------------------------------------------------------------------------

def _real_get(url, headers, timeout, max_bytes):
    """(status, body, error) of one HTTPS GET; no redirect to another host is followed."""
    import urllib.error
    import urllib.request

    class _Same(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, hdrs, newurl):
            if urlsplit(newurl).hostname != urlsplit(req.full_url).hostname:
                return None
            return super().redirect_request(req, fp, code, msg, hdrs, newurl)
    try:
        req = urllib.request.Request(url, headers=dict(headers), method="GET")
        with urllib.request.build_opener(_Same).open(req, timeout=timeout) as r:
            body = r.read(max_bytes + 1)
            return r.status, body, ""
    except urllib.error.HTTPError as exc:
        return exc.code, b"", f"HTTP {exc.code}"
    except Exception as exc:                        # timeout, DNS, TLS: a failed call
        return 0, b"", type(exc).__name__


def _real_run(argv, timeout, env=None):
    return skillsfresh._raw_run(argv, timeout, env)


class Tools:
    """The network and process access of one run. `get(url, headers, timeout, max_bytes) -> (status, body, err)` and
    `run(argv, timeout, env) -> (rc, out, err)` are replaceable. `network` calls fail at once offline or when the budget is
    spent; their timeout is the smaller of CALL_TIMEOUT_S and the budget left."""

    def __init__(self, get: Callable | None = None, run: Callable | None = None, *, offline: bool = False,
                 budget_s: float = BUDGET_S, clock: Callable = time.monotonic, environ: dict | None = None,
                 which: Callable | None = None):
        import shutil
        self.get, self.run = get or _real_get, run or _real_run
        self.which = which or (lambda name: shutil.which(name) is not None)
        self.offline, self.budget_s, self.clock = offline, budget_s, clock
        self.environ = dict(os.environ if environ is None else environ)
        self.started: float | None = None
        self.calls = 0

    def _timeout(self) -> tuple[float | None, str]:
        if self.offline:
            return None, "offline: no network call was made"
        if self.started is None:
            self.started = self.clock()
        left = self.budget_s - (self.clock() - self.started)
        if left <= 0:
            return None, "the network budget for this run is spent"
        return min(CALL_TIMEOUT_S, left), ""

    def json(self, url: str):
        """(value, why). `value` is the parsed JSON, `why` is "" on success."""
        status, body, why = self.text(url, as_bytes=True)
        if why:
            return None, why
        try:
            return safeio.parse_json_bytes(body, max_bytes=BODY_MAX), ""
        except safeio.SafeReadError as exc:
            return None, f"the answer was not usable JSON ({type(exc).__name__})"

    def text(self, url: str, as_bytes: bool = False):
        host = urlsplit(url)
        if host.scheme != "https" or host.hostname not in HOSTS:
            raise NotReadOnly(f"{clean(url, 80)} is not an allowed https host")
        timeout, why = self._timeout()
        if timeout is None:
            return 0, b"", why
        headers = {"User-Agent": "tt-orchard-updates", "Accept": "application/json, text/html"}
        token = self.environ.get("HF_TOKEN")
        if token and host.hostname == "huggingface.co":     # read from the environment only; sent to one host
            headers["Authorization"] = f"Bearer {token}"
        self.calls += 1
        try:
            status, body, err = self.get(url, headers, timeout, BODY_MAX)
        except Exception as exc:
            return 0, b"", f"{type(exc).__name__}"
        if err or status != 200:
            code = f"HTTP {status}" if status else ""
            return status, b"", clean(err or code or "no answer", 60)
        if not isinstance(body, (bytes, bytearray)) or len(body) > BODY_MAX:
            return status, b"", "the answer was larger than the limit"
        return status, (bytes(body) if as_bytes else bytes(body).decode("utf-8", "replace")), ""

    def cmd(self, argv) -> tuple[int, str, str]:
        """Run `<tool> --version` or a read-only git command. Anything else raises NotReadOnly."""
        argv = [str(a) for a in argv]
        prog = os.path.basename(argv[0]) if argv else ""
        if prog in VERSION_TOOLS:
            if argv[1:] != ["--version"]:
                raise NotReadOnly(f"{prog} may only be run as `{prog} --version`")
        elif prog == "git":
            sub, rest = skillsfresh._git_subcommand(argv)
            if sub in EXTRA_GIT:
                if "--output" in rest or any(a.startswith("--output=") for a in rest):
                    raise NotReadOnly("git --output writes a file")
            else:
                try:
                    skillsfresh.refuse_unless_read_only(argv)
                except skillsfresh.NotReadOnly as exc:
                    raise NotReadOnly(str(exc)) from None
        else:
            raise NotReadOnly(f"{prog or 'an empty command'} is not allowed")
        env = dict(self.environ, GIT_TERMINAL_PROMPT="0", GIT_OPTIONAL_LOCKS="0", LC_ALL="C")
        try:
            return self.run(argv, CALL_TIMEOUT_S, env)
        except Exception as exc:
            return 127, "", type(exc).__name__


# ---- rows -------------------------------------------------------------------------------------------------------

@dataclass
class Row:
    key: str                    # the rule-table name: tt-lang, tt-metal, tt-model-manager, tt-cli, tt-gozer, coder, skills
    item: str
    installed: str = "-"
    installed_date: str | None = None
    upstream: str = "-"
    upstream_date: str | None = None
    status: str = UNKNOWN
    why: str = ""               # the reason for an unknown row, or a note on the others
    proposal: list = field(default_factory=list)    # commands as text; nothing here is ever run
    moves_with: list = field(default_factory=list)
    ready: str = "-"

    def label(self) -> str:
        return f"unknown ({self.why})" if self.status == UNKNOWN else self.status


def _parse_date(s) -> datetime | None:
    if not isinstance(s, str):
        return None
    try:
        d = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def _date(s) -> str | None:
    d = _parse_date(clean(s, 40))
    return d.date().isoformat() if d else None


def age_text(date: str | None, now: float) -> str:
    d = _parse_date(date)
    if d is None:
        return "?"
    days = int((now - d.timestamp()) // 86400)
    return "today" if days <= 0 else f"{days} d"


_VER = re.compile(r"\d+(?:\.\d+)+")


def vtuple(s) -> tuple[int, ...] | None:
    m = _VER.search(s) if isinstance(s, str) else None
    if not m:
        m2 = re.search(r"\d+", s) if isinstance(s, str) else None
        return (int(m2.group()),) if m2 else None
    return tuple(int(p) for p in m.group().split(".")[:6])


def _github_latest(tools: Tools, repo: str):
    """(tag, sha, date, why) of the latest release, else the newest tag. The repo name is checked before use."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo or ""):
        return None, None, None, "the repository name in the config is not owner/name"
    rel, why = tools.json(f"https://api.github.com/repos/{repo}/releases/latest")
    if isinstance(rel, dict) and isinstance(rel.get("tag_name"), str):
        return clean(rel["tag_name"], 60), None, _date(rel.get("published_at")), ""
    tags, why2 = tools.json(f"https://api.github.com/repos/{repo}/tags?per_page=30")
    if isinstance(tags, list):
        best = _newest_tag(tags)
        if best:
            return best[0], best[1], None, ""
        return None, None, None, f"{repo} has no release or tag with a version"
    return None, None, None, why2 or why or "no answer"


def _newest_tag(tags: list):
    best = None
    for t in tags[:LIST_MAX]:
        if not isinstance(t, dict) or not isinstance(t.get("name"), str):
            continue
        key = vtuple(t["name"])
        if key is None or "dev" in t["name"] or "rc" in t["name"]:
            continue
        sha = (t.get("commit") or {}).get("sha") if isinstance(t.get("commit"), dict) else None
        if best is None or key > best[2]:
            best = (clean(t["name"], 60), clean(sha, 40) if isinstance(sha, str) else None, key)
    return best


def _commit_date(tools: Tools, repo: str, ref: str):
    c, why = tools.json(f"https://api.github.com/repos/{repo}/commits/{quote(ref, safe='')}")
    if isinstance(c, dict) and isinstance(c.get("commit"), dict):
        who = c["commit"].get("committer")
        return _date(who.get("date")) if isinstance(who, dict) else None
    return None


# ---- the checks -------------------------------------------------------------------------------------------------

def check_tool(tools: Tools, cfg: dict, key: str, argv0: str, label: str, upgrade: str) -> Row:
    row = Row(key, label)
    if not tools.which(argv0):
        row.why = f"{argv0} is not installed"
        return row
    rc, out, err = tools.cmd([argv0, "--version"])
    local = vtuple(clean(out, 200))
    if rc != 0 or local is None:
        row.why = f"{argv0} --version gave no version"
        return row
    row.installed = ".".join(map(str, local))
    tag, _sha, date, why = _github_latest(tools, cfg["repos"].get(key, ""))
    if tag is None:
        row.why = f"upstream: {why}"
        return row
    row.upstream, row.upstream_date = tag, date
    up = vtuple(tag)
    if up is None:
        row.why = "the upstream tag has no version number"
    elif up > local:
        row.status = BEHIND
        row.proposal = [upgrade]
    else:
        row.status = CURRENT
    return row


def _installed_index(path: Path) -> dict:
    try:
        data = safeio.read_json(path, max_bytes=4 * 1024 * 1024, expect=dict)
    except safeio.SafeReadError:
        return {}
    return data


def coder_packages(cfg: dict) -> list[str]:
    if cfg["coder_packages"]:
        return list(cfg["coder_packages"])
    from orchard import setup_machine
    return [v["package"] for k, v in setup_machine.CODERS.items() if "coder" in k]


def check_coders(tools: Tools, cfg: dict, installed_path: Path) -> list[Row]:
    rows, index = [], _installed_index(installed_path)
    pkgs = coder_packages(cfg)
    for pkg in pkgs:
        row = Row("coder", f"coder {pkg.split('/')[-1]}")
        ent = index.get(pkg)
        rev = ent.get("revision") if isinstance(ent, dict) else None
        if not isinstance(rev, str) or not re.fullmatch(r"[0-9a-f]{7,64}", rev):
            row.why = "not installed here (tt-model list)" if ent is None else "the install record has no revision"
        else:
            row.installed = rev[:10]
        info, why = tools.json(f"https://huggingface.co/api/models/{quote(pkg, safe='/')}")
        sha = info.get("sha") if isinstance(info, dict) else None
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{7,64}", sha):
            row.why = row.why or f"Hugging Face: {why or 'no commit sha in the answer'}"
            rows.append(row)
            continue
        row.upstream, row.upstream_date = sha[:10], _date(info.get("lastModified"))
        if row.installed != "-":
            old, _ = tools.json(f"https://huggingface.co/api/models/{quote(pkg, safe='/')}/revision/{rev}")
            row.installed_date = _date(old.get("lastModified")) if isinstance(old, dict) else None
            if sha.startswith(rev) or rev.startswith(sha):
                row.status = CURRENT
            else:
                row.status = BEHIND
                row.proposal = [f"tt-model pull {pkg}", "tt-orchard coder stop --yes   # then start the coder again; "
                                "this restarts the container, so run it when no run needs the coder"]
        rows.append(row)
    rows.append(_candidates(tools, cfg, pkgs))
    return rows


def _candidates(tools: Tools, cfg: dict, pkgs: list[str]) -> Row:
    row = Row("coder", "coder candidates", installed=", ".join(p.split("/")[-1] for p in pkgs) or "-")
    fam = [f.lower() for f in cfg["family"] if f]
    authors = cfg["authors"] or sorted({p.split("/")[0] for p in pkgs})
    urls = [f"https://huggingface.co/api/models?author={quote(a, safe='')}&sort=lastModified&direction=-1&limit=100"
            for a in authors]
    urls += [f"https://huggingface.co/api/models?search={quote(f, safe='')}&sort=lastModified&direction=-1&limit=50"
             for f in fam]
    found, errors, answered = {}, [], 0
    for u in urls:
        lst, why = tools.json(u)
        if not isinstance(lst, list):
            errors.append(why or "not a list")
            continue
        answered += 1
        for m in lst[:LIST_MAX]:
            mid = m.get("id") if isinstance(m, dict) else None
            if not isinstance(mid, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", mid):
                continue
            if mid in pkgs or not any(f in mid.lower() for f in fam):
                continue
            d = _date(m.get("lastModified")) or _date(m.get("createdAt")) or "?"
            found[mid] = max(found.get(mid, ""), d)
    if not answered:
        row.why = f"Hugging Face: {errors[0] if errors else 'no answer'}"
        return row
    if found:
        top = sorted(found.items(), key=lambda kv: kv[1], reverse=True)[:5]
        row.status = CANDIDATE
        row.upstream = "; ".join(f"{k} ({d})" for k, d in top)
        row.upstream_date = top[0][1] if top[0][1] != "?" else None
        row.why = f"{len(found)} repo(s) match {', '.join(fam)} and are not configured"
        row.proposal = [f"tt-model info {top[0][0]}   # read its manifest; configure it only after the kernel probe"]
    else:
        row.status = CURRENT
        row.upstream = "none"
    return row


_WHEEL = re.compile(r"tt_lang(?:_light)?-(\d+(?:\.\d+)*)(?:\.dev(\d{8}))?")


def newest_dev_wheel(html: str):
    """(version text, release tuple, dev date) of the newest tt_lang wheel named in the index page."""
    best = None
    for m in _WHEEL.finditer(html[:BODY_MAX]):
        rel = tuple(int(p) for p in m.group(1).split(".")[:6])
        key = (rel, m.group(2) or "99999999")
        if best is None or key > best[0]:
            best = (key, m.group(1) + (f".dev{m.group(2)}" if m.group(2) else ""))
    return (best[1], best[0][0], best[0][1]) if best else None


def check_ttlang(tools: Tools, cfg: dict, ver: verifications.Verifications) -> Row:
    pin = ttlang_pin.SHA
    repo = cfg["repos"]["tt-lang"]
    row = Row("tt-lang", "tt-lang", installed=f"{ttlang_pin.VERSION_TAG} {pin[:9]} (the pin)")
    row.installed_date = _commit_date(tools, repo, pin)
    tags, why = tools.json(f"https://api.github.com/repos/{repo}/tags?per_page=30")
    best = _newest_tag(tags) if isinstance(tags, list) else None
    html_status, html, why2 = tools.text(DEV_INDEX)
    if not why2:
        # The root page links one directory per month (2026-10/); the wheels are listed in the newest month.
        months = sorted(set(re.findall(r'href="(\d{4}-\d{2})/"', html[:BODY_MAX])))
        if months:
            _s, html, why2 = tools.text(f"{DEV_INDEX}{months[-1]}/")
    dev = newest_dev_wheel(html) if not why2 else None
    pypi, why3 = tools.json("https://pypi.org/pypi/tt-lang/json")
    pv = clean((pypi.get("info") or {}).get("version"), 40) if isinstance(pypi, dict) and isinstance(pypi.get("info"), dict) else ""
    parts = []
    if best:
        parts.append(f"tag {best[0]}")
    if dev:
        parts.append(f"dev wheel {dev[0]}")
    if pv:
        parts.append(f"PyPI {pv}")
    row.upstream = ", ".join(parts) or "-"
    if not parts:
        row.why = f"upstream: {why or why2 or why3 or 'no answer'}"
        return row
    pin_t = vtuple(ttlang_pin.VERSION_TAG)
    verified_wheels = sorted(r.versions["tt-lang-wheel"] for r in ver.records if r.result == "pass"
                             and r.versions.get("tt-lang") == pin and "tt-lang-wheel" in r.versions)
    if best and best[2] > pin_t:
        row.status = BEHIND
        row.upstream_date = _commit_date(tools, repo, best[1]) if best[1] else None
        row.proposal = [f"# find a tt-lang commit that builds on your tt-metal ref: {FIND_COMPATIBLE}",
                        f"# then set SHA and VERSION_TAG in orchard/ttlang_pin.py to {best[0]} and run:",
                        "tt-orchard kernel-env", "tt-orchard ttlang-consistency"]
    elif dev and (dev[1] > pin_t or (verified_wheels and dev[0] not in verified_wheels
                                     and re.sub(r".*dev", "", dev[0]) > re.sub(r".*dev", "", verified_wheels[-1]))):
        row.status = CANDIDATE
        row.proposal = [f"# a newer dev wheel exists on the S3 index ({dev[0]}); read its build_info, then pair it with a tt-metal ref",
                        "tt-orchard ttlang-consistency"]
    else:
        row.status = CURRENT
    if row.status == CURRENT and not pv:
        row.why = "public PyPI did not answer"
    return row


def check_ttmetal(tools: Tools, cfg: dict, tree: Path) -> Row:
    row = Row("tt-metal", "tt-metal (host tree)")
    if not tree.is_dir():
        row.why = f"no tt-metal tree at {tree.name} (use --tt-metal-dir or ORCHARD_TT_METAL_DIR)"
        return row
    rc, out, _ = tools.cmd(["git", "-C", str(tree), "describe", "--tags", "--always"])
    desc = clean(out.strip(), 80)
    if rc != 0 or not desc:
        row.why = "git describe failed on the tree"
        return row
    row.installed = desc
    rc, out, _ = tools.cmd(["git", "-C", str(tree), "show", "-s", "--format=%cI", "HEAD"])
    row.installed_date = _date(out.strip()) if rc == 0 else None
    tag, _sha, date, why = _github_latest(tools, cfg["repos"]["tt-metal"])
    if tag is None:
        row.why = f"upstream: {why}"
        return row
    row.upstream, row.upstream_date = tag, date
    local, up = vtuple(desc), vtuple(tag)
    if local is None or up is None:
        row.why = "no version number to compare"
    elif up > local:
        row.status = BEHIND
        row.proposal = [f"git -C <tree> fetch --tags && git -C <tree> checkout {tag}   # then rebuild the tree and ttnn",
                        "# before that, find the tt-lang commit that builds on it (rule R1)"]
    else:
        row.status = CURRENT
    return row


def check_skills(results) -> Row:
    row = Row("skills", "tenstorrent/skills checkout")
    if not results:
        row.why = "no skills_dirs in the bringup config"
        return row
    worst, notes = CURRENT, []
    for r in results:
        notes.append(clean(f"{r.path}: {r.state}", 200))
        if r.state in (skillsfresh.BEHIND, skillsfresh.DIFFERS):
            worst = BEHIND
            row.proposal.append(clean(r.update_text(), 300))
        elif r.state == skillsfresh.UNKNOWN and worst == CURRENT:
            worst = UNKNOWN
            row.why = clean(r.detail, 120)
    row.status = worst
    row.installed = "; ".join(notes)
    row.upstream = "main"
    return row


def with_rules(rows: list[Row], ver: verifications.Verifications, combo: dict) -> list[Row]:
    """Fill `moves_with` for rows that need an upgrade, and `ready` for every row."""
    by_key = {r.key: r for r in rows}
    for r in rows:
        if r.status in (BEHIND, CANDIDATE):
            for rule in RULES:
                if r.key in rule["members"]:
                    others = [m for m in rule["members"] if m != r.key]
                    r.moves_with.append(f"{rule['id']}: moves with {', '.join(others)}. {rule['text']}")
        if r.status in (BEHIND, CANDIDATE):
            rec = ver.match(dict(combo, **{r.key: r.upstream})) if r.key in combo else None
            if rec and rec.result == "pass":
                r.ready = f"verified ({rec.date}, {rec.evidence})"
            else:
                r.ready = ("unverified: run the smoke and probe kernels on the pair (see "
                           "docs/run-logs/2026-10-29-ttlang-pair-chip-check.md)")
        elif r.status == CURRENT:
            r.ready = "in use"
    return rows


def run_all(tools: Tools, cfg: dict, *, installed_path: Path, tt_metal_dir: Path, skills: list | None,
            ver: verifications.Verifications) -> list[Row]:
    """Every tracked item. A check that raises anything but NotReadOnly becomes an unknown row; NotReadOnly is a bug and
    propagates so a test sees it."""
    def guard(key, label, fn):
        try:
            out = fn()
            return out if isinstance(out, list) else [out]
        except NotReadOnly:
            raise
        except Exception as exc:
            return [Row(key, label, why=f"check failed: {type(exc).__name__}")]
    rows: list[Row] = []
    rows += guard("coder", "coder", lambda: check_coders(tools, cfg, installed_path))
    rows += guard("tt-lang", "tt-lang", lambda: check_ttlang(tools, cfg, ver))
    rows += guard("tt-metal", "tt-metal", lambda: check_ttmetal(tools, cfg, tt_metal_dir))
    rows += guard("tt-model-manager", "tt-model-manager", lambda: check_tool(
        tools, cfg, "tt-model-manager", "tt-model", "tt-model-manager (tt-model)", "uv pip install -U tt-model-manager   # or the install route you used"))
    rows += guard("tt-cli", "tt-cli", lambda: check_tool(
        tools, cfg, "tt-cli", "tt", "tt-cli (tt, the official CLI)", "update tt-cli from its release page; do not install a tool named tt of your own"))
    rows += guard("tt-gozer", "tt-gozer", lambda: check_tool(
        tools, cfg, "tt-gozer", "gozer", "tt-gozer (gozer)", "git -C <tt-gozer checkout> pull --ff-only   # then reinstall as its README says"))
    rows += guard("skills", "skills", lambda: check_skills(skills) if skills is not None else Row("skills", "tenstorrent/skills checkout", why="no skills check supplied"))
    combo = {"tt-lang": ttlang_pin.SHA, "tt-metal": next((r.installed.split("-")[0] for r in rows if r.key == "tt-metal" and r.installed != "-"), "")}
    return with_rules(rows, ver, combo)


# ---- cache, report, hint ----------------------------------------------------------------------------------------

def cache_path() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "tt-orchard" / "updates.json"


def render(rows: list[Row], now: float, *, verbose: bool = True) -> list[str]:
    lines = [f"{'item':36} {'installed':26} {'age':7} {'newest upstream':44} {'age':7} status"]
    for r in rows:
        lines.append(f"{clean(r.item, 36):36} {clean(r.installed, 26):26} {age_text(r.installed_date, now):7} "
                     f"{clean(r.upstream, 44):44} {age_text(r.upstream_date, now):7} {r.label()}")
    for r in rows:
        if r.status in (BEHIND, CANDIDATE):
            lines += ["", f"{r.item}: {r.status}. {clean(r.why, 200)}".rstrip()]
            lines += [f"  moves with: {m}" for m in r.moves_with]
            lines += [f"  ready to farm: {r.ready}"]
            lines += [f"  propose (text only, orchard runs none of it): {c}" for c in r.proposal]
    lines += ["", "This command changes nothing. Run any proposed command yourself."]
    return lines


def hint_from_cache(path: Path | None = None) -> str | None:
    """One line for preflight, from the cache only. None when the cache is missing or nothing needs attention."""
    cache = skillsfresh.Cache(path or cache_path(), write=False)
    ent = cache.get("report")
    rows = ent.get("rows") if ent else None
    if not isinstance(rows, list):
        return None
    behind = [r.get("item") for r in rows if isinstance(r, dict) and r.get("status") == BEHIND]
    cand = [r for r in rows if isinstance(r, dict) and r.get("status") == CANDIDATE]
    if not behind and not cand:
        return None
    bits = []
    if behind:
        bits.append("behind: " + ", ".join(clean(b, 40) for b in behind[:6]))
    if cand:
        bits.append("newer candidate: " + ", ".join(clean(c.get("item"), 40) for c in cand[:3]))
    return f"updates ({'; '.join(bits)}). Run `tt-orchard updates` for details."


def main(args, *, out=None, tools: Tools | None = None, skills=None, now: float | None = None) -> int:
    say = (lambda s: print(s, file=out)) if out is not None else print
    try:
        cfg = load_config(getattr(args, "updates_config", None))
    except ConfigError as exc:
        say(f"updates config: {exc}")
        return 2
    now = time.time() if now is None else now
    cache = skillsfresh.Cache(cache_path())
    offline = bool(getattr(args, "offline", False))
    ent = cache.get("report")
    age = cache.age_s(ent) if ent else None
    use_cache = (ent is not None and age is not None and 0 <= age <= cfg["cache_hours"] * 3600
                 and not getattr(args, "refresh", False) and not offline and isinstance(ent.get("rows"), list))
    note = ""
    if use_cache or (offline and ent and isinstance(ent.get("rows"), list)):
        rows = [Row(**{k: v for k, v in r.items() if k in Row.__dataclass_fields__}) for r in ent["rows"] if isinstance(r, dict)]
        note = f"from the cache, checked {skillsfresh._age_text(age)} ago" if age is not None else "from the cache"
    else:
        tools = tools or Tools(offline=offline)
        ver_path = verifications.default_path()
        ver = verifications.load(ver_path)
        home = Path.home()
        tree = Path(getattr(args, "tt_metal_dir", None) or os.environ.get("ORCHARD_TT_METAL_DIR") or home / "tt-metal")
        if skills is None and getattr(args, "skills_dirs", None):
            skills = skillsfresh.check_skills_dirs(args.skills_dirs, offline=offline,
                                                   cache=skillsfresh.Cache(write=False))
        rows = run_all(tools, cfg, installed_path=home / ".cache" / "tt-model" / "installed.json", tt_metal_dir=tree,
                       skills=skills, ver=ver)
        for p in ver.problems:
            say(f"verifications: {clean(p, 200)}")
        if any(r.status != UNKNOWN for r in rows):
            cache.put("report", {"rows": [asdict(r) for r in rows]})
    if getattr(args, "json", False):
        say(json.dumps({"checked_at": now, "note": note, "rows": [dict(asdict(r), label=r.label()) for r in rows]}, indent=2))
    else:
        if note:
            say(note)
        for line in render(rows, now):
            say(line)
    return 0
