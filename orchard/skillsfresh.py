# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Is each skill source the latest one upstream has? (spec 2026-10-15-kernel-attempt-design.md, section 11, item 28)

The operator's rule (2026-10-16): "WE USE THE LATEST tenstorrent/skills and tt-lang skills available. We check in
preflight whether we have the latest even." This module does the checking. It reads and never changes anything:
orchard does not update a clone or a skill itself. Each result names the command a person runs to update.

Three sources:

- SOURCE_SKILLS: the tenstorrent/skills clone(s) that bringup.toml's `skills_dirs` point into. HEAD is compared with
  `git ls-remote origin refs/heads/main`. A dirty tree or a branch other than main is a warning that names it.
- SOURCE_TTLANG: the tt-lang source checkout that `kernel-env` builds the simulator from. Its HEAD is the pin
  (defaults.TTLANG_PIN), so "behind" here means upstream main has moved past the pin.
- SOURCE_TTLANG_SKILL: the installed tt-lang skill (default ~/.claude/skills/ttm-tt-lang). tenstorrent/skills has no
  tt-lang skill. This one was vendored by tt-warp (`tt-warp fastmodels sync`) from tenstorrent/tt-metal, branch
  agentic-research/fast-models-fast, path .agents/skills/tt-lang; tt-metal's main branch removed that path, so the
  branch is the only upstream copy. Each file's git blob sha is compared with the branch's tree, read with one
  `gh api` call. SKILL.md is compared after the installer's two edits are undone (upstream_skill_md).

States: CURRENT, BEHIND (with a count), AHEAD, DIFFERS (not equal, and no count could be made) and UNKNOWN (the
comparison could not be made: no network, no origin on github.com, no checkout, a malformed answer). UNKNOWN never
claims current. A result is `stale` when it is BEHIND or DIFFERS, or UNKNOWN without a recent check that found it
current: a cached CURRENT result for the same local commit (or the same local file shas) younger than
CACHE_MAX_AGE_S. The cache is `~/.cache/tt-orchard/skills-freshness.json` unless a caller names another file.

Safety: every command goes through ReadOnlyRunner, which refuses anything but a short list of read-only git
subcommands and `gh api` without a method or field option. `git status` runs with GIT_OPTIONAL_LOCKS=0, so it does not
refresh (write) the index. Network calls go only to github.com: an origin elsewhere is UNKNOWN and is not contacted,
and gh is pointed at github.com with GH_HOST, so its token goes nowhere else. Each network call has a timeout, and a
run has a network budget (BUDGET_S); after it, the remaining calls fail at once and their sources are UNKNOWN.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

CURRENT, BEHIND, AHEAD, DIFFERS, UNKNOWN = "current", "behind", "ahead", "differs", "unknown"
SOURCE_SKILLS = "tenstorrent/skills"
SOURCE_TTLANG = "tt-lang source"
SOURCE_TTLANG_SKILL = "tt-lang skill"

CALL_TIMEOUT_S = 15.0           # choice: one ls-remote or gh call; a healthy one takes about a second
BUDGET_S = 45.0                 # choice: all network calls of one check together
CACHE_MAX_AGE_S = 24 * 3600.0   # choice: how old a CURRENT result may be and still cover a failed comparison
FILE_MAX_BYTES = 4 * 1024 * 1024    # choice: the largest skill file is 59 KB; anything bigger is not a skill file

# The installed tt-lang skill's upstream (read 2026-10-16).
TTLANG_SKILL_REPO = "tenstorrent/tt-metal"
TTLANG_SKILL_REF = "agentic-research/fast-models-fast"
TTLANG_SKILL_PATH = ".agents/skills/tt-lang"
TTLANG_SKILL_UPSTREAM_NAME = "tt-lang"
TTLANG_SKILL_UPDATE = ("update the tt-metal checkout tt-warp copies from, then copy again: "
                       "`git -C ~/code/tt-metal-fastmodels pull --ff-only` and "
                       "`tt-warp fastmodels sync --from ~/code/tt-metal-fastmodels`")
OVERRIDE = "--allow-stale-skills"

# git subcommands that only read. `remote` is allowed only as `remote get-url`.
READ_ONLY_GIT = frozenset({"rev-parse", "symbolic-ref", "status", "remote", "ls-remote", "cat-file", "rev-list"})
GH_WRITE_OPTIONS = frozenset({"-X", "--method", "-f", "-F", "--field", "--raw-field", "--input"})
GIT_SAFE_CONFIG = (("http.followRedirects", "false"), ("core.fsmonitor", "false"), ("core.hooksPath", "/dev/null"))
# Options that take a value and come before a git subcommand.
_GIT_GLOBAL_WITH_VALUE = frozenset({"-C", "-c", "--git-dir", "--work-tree"})


class NotReadOnly(Exception):
    """A command that could change something reached the runner. It is never run."""


def default_cache_path() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "tt-orchard" / "skills-freshness.json"


def _raw_run(argv, timeout, env=None):
    """(returncode, stdout, stderr), like ttlang_env.real_run. A timeout answers 124 and a missing program 127.

    The command runs in its own session. On a timeout the whole process group is killed, so a helper git started
    (git-remote-https, ssh) does not outlive the call; subprocess.run would kill only the direct child."""
    import signal
    import subprocess
    try:
        proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, env=env, start_new_session=True)
    except (OSError, subprocess.SubprocessError) as exc:
        return 127, "", f"{type(exc).__name__}: {exc}"
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        if proc.pid > 1:                                # its own group (start_new_session), never ours
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        proc.communicate()
        return 124, "", f"timed out after {timeout:.0f} s"
    return proc.returncode, out, err


def _git_subcommand(argv) -> tuple[str | None, list[str]]:
    """The subcommand of a git argv and the arguments after it."""
    i = 1
    while i < len(argv):
        a = argv[i]
        if a in _GIT_GLOBAL_WITH_VALUE:
            i += 2
            continue
        if a.startswith("-"):
            i += 1
            continue
        return a, list(argv[i + 1:])
    return None, []


def refuse_unless_read_only(argv) -> None:
    """Raise NotReadOnly unless argv is a read-only git call or a `gh api` GET."""
    if not argv:
        raise NotReadOnly("empty command")
    prog = os.path.basename(str(argv[0]))
    if prog == "git":
        sub, rest = _git_subcommand(argv)
        if sub not in READ_ONLY_GIT:
            raise NotReadOnly(f"git {sub} is not a read-only command")
        if sub == "remote" and rest[:1] != ["get-url"]:
            raise NotReadOnly("only `git remote get-url` is allowed")
        return
    if prog == "gh":
        if argv[1:2] != ["api"]:
            raise NotReadOnly("only `gh api` is allowed")
        if any(a in GH_WRITE_OPTIONS or a.split("=", 1)[0] in GH_WRITE_OPTIONS for a in argv[2:]):
            raise NotReadOnly("`gh api` with a method or a field could write")
        return
    raise NotReadOnly(f"{prog} is not git or gh")


class ReadOnlyRunner:
    """Runs read-only git and gh commands with a quiet environment, a timeout per call and a network budget.

    `run` is (argv, timeout, env) -> (returncode, stdout, stderr). Every argv passes refuse_unless_read_only first.
    A call marked `network` fails at once (no process starts) when the runner is offline or the budget is spent; its
    timeout is the smaller of CALL_TIMEOUT_S and the budget left. An exception from `run` is a failure (127)."""

    def __init__(self, run: Callable | None = None, *, budget_s: float = BUDGET_S, offline: bool = False,
                 clock: Callable = time.monotonic, environ: dict | None = None):
        # _raw_run is looked up at each construction, so the test suite's guard (conftest.py) can replace it.
        self.run, self.budget_s, self.offline, self.clock = run or _raw_run, budget_s, offline, clock
        self.started: float | None = None
        base = dict(os.environ if environ is None else environ)
        base.update({"GIT_TERMINAL_PROMPT": "0", "GIT_OPTIONAL_LOCKS": "0", "GH_HOST": "github.com",
                     "GH_PROMPT_DISABLED": "1", "GH_NO_UPDATE_NOTIFIER": "1", "LC_ALL": "C"})
        base.setdefault("GIT_SSH_COMMAND", "ssh -o BatchMode=yes -o ConnectTimeout=10")
        # Configuration for every git call, added after any GIT_CONFIG_* entries the operator has: follow no HTTP
        # redirect away from github.com, and run no fsmonitor command or hook (a clone's own configuration could name
        # one, and `git status` would run it).
        try:
            n = max(0, int(base.get("GIT_CONFIG_COUNT", "0")))
        except ValueError:
            n = 0
        for key, value in GIT_SAFE_CONFIG:
            base[f"GIT_CONFIG_KEY_{n}"], base[f"GIT_CONFIG_VALUE_{n}"] = key, value
            n += 1
        base["GIT_CONFIG_COUNT"] = str(n)
        self.env = base

    def __call__(self, argv, *, network: bool) -> tuple[int, str, str]:
        refuse_unless_read_only(argv)
        timeout = CALL_TIMEOUT_S
        if network:
            if self.offline:
                return 124, "", "offline: no network call was made"
            if self.started is None:
                self.started = self.clock()
            left = self.budget_s - (self.clock() - self.started)
            if left <= 0:
                return 124, "", "the network budget for this check is spent"
            timeout = min(CALL_TIMEOUT_S, left)
        try:
            return self.run(list(argv), timeout, self.env)
        except Exception as exc:                        # a broken runner is a failed call, never a crash
            return 127, "", f"{type(exc).__name__}: {exc}"


class Cache:
    """The last result of each source: {key: {"checked_at", "local", "remote", "state", ...}}.

    A broken or unreadable file gives an empty cache. A write replaces the file whole (a temporary file and a
    rename), so a link at the path is itself replaced and its target is never written. `write=False` keeps everything in memory (kernel-env --check
    changes nothing on the machine)."""

    def __init__(self, path: Path | None = None, *, clock: Callable = time.time, write: bool = True):
        self.path = Path(path) if path is not None else default_cache_path()
        self.clock, self.write = clock, write
        self.data = self._load()

    def _load(self) -> dict:
        try:
            st = os.lstat(self.path)
            if not stat.S_ISREG(st.st_mode) or st.st_size > FILE_MAX_BYTES:
                return {}
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, RecursionError):      # RecursionError: a deeply nested file
            return {}
        return data if isinstance(data, dict) else {}

    def get(self, key: str) -> dict | None:
        entry = self.data.get(key)
        return entry if isinstance(entry, dict) else None

    def put(self, key: str, entry: dict) -> None:
        self.data[key] = dict(entry, checked_at=self.clock())
        if not self.write:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".skills-freshness-")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, sort_keys=True)
            os.replace(tmp, self.path)
        except OSError:
            pass                                        # the cache is a convenience; a failed write changes no result

    def age_s(self, entry: dict) -> float | None:
        t = entry.get("checked_at")
        return self.clock() - t if isinstance(t, (int, float)) and not isinstance(t, bool) else None


@dataclass
class Freshness:
    source: str
    path: str
    state: str
    detail: str
    local: str | None = None            # HEAD sha, or a digest of the local file shas
    remote: str | None = None           # upstream main's sha, or upstream's tree as text
    behind: int | None = None
    ahead: int | None = None
    branch: str | None = None
    dirty: bool | None = None
    update: str | None = None           # the command a person runs to update; orchard never runs it
    warnings: list[str] = field(default_factory=list)
    cached: dict | None = None          # an UNKNOWN result's last good check, when one was used

    @property
    def stale(self) -> bool:
        if self.state in (BEHIND, DIFFERS):
            return True
        return self.state == UNKNOWN and not (self.cached and self.cached.get("state") == CURRENT)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["stale"] = self.stale
        return out

    def update_text(self) -> str:
        """What to do about a stale result: the update command for BEHIND or DIFFERS, and for any other state (an
        UNKNOWN comparison) the advice to compare again. An UNKNOWN result never states an update, whatever is
        attached to it, because no comparison found anything to update."""
        if self.state in (BEHIND, DIFFERS) and self.update:
            return f"To update: {self.update}."
        return "Compare again when GitHub answers."

    def line(self) -> str:
        """One line for a terminal or a report."""
        text = f"{self.source} ({self.path}): {self.state}. {self.detail}"
        for w in self.warnings:
            text += f" Warning: {w}"
        return text


def stale(results) -> list[Freshness]:
    return [r for r in results if r.stale]


def _age_text(seconds: float) -> str:
    if seconds < 120:
        return f"{seconds:.0f} s"
    if seconds < 7200:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def _unknown(result: Freshness, why: str, cache: Cache, key: str) -> Freshness:
    """Mark `result` UNKNOWN. A cached CURRENT result for the same local state, younger than CACHE_MAX_AGE_S, is
    attached (and then the result is not stale); the detail always says the comparison was not made."""
    result.state = UNKNOWN
    result.update = None                    # nothing was found that an update would fix
    result.detail = f"It could not be compared with upstream ({why})."
    entry = cache.get(key)
    age = cache.age_s(entry) if entry else None
    if (entry and entry.get("state") == CURRENT and entry.get("local") == result.local and age is not None
            and 0 <= age <= CACHE_MAX_AGE_S):
        result.cached = entry
        result.detail += f" A check {_age_text(age)} ago found this same local state current (last seen current)."
    return result


def _reused(result: Freshness, cache: Cache, key: str, reuse_s: float) -> bool:
    """Fill `result` from a cached check of the same local state younger than `reuse_s`, without the network."""
    if reuse_s <= 0:
        return False
    entry = cache.get(key)
    age = cache.age_s(entry) if entry else None
    if not entry or entry.get("local") != result.local or age is None or not 0 <= age <= reuse_s:
        return False
    if entry.get("state") not in (CURRENT, BEHIND, AHEAD, DIFFERS):
        return False
    result.state, result.remote = entry["state"], entry.get("remote")
    result.behind, result.ahead = entry.get("behind"), entry.get("ahead")
    result.detail = f"{entry.get('detail') or ''} (checked {_age_text(age)} ago)".strip()
    result.update = entry.get("update")
    return True


def _remember(result: Freshness, cache: Cache, key: str) -> None:
    cache.put(key, {"state": result.state, "local": result.local, "remote": result.remote, "behind": result.behind,
                    "ahead": result.ahead, "detail": result.detail, "update": result.update})


# ---- git clones -------------------------------------------------------------------------------------------------

_GH_SCP = re.compile(r"^git@github\.com:/?(?P<path>.+)$")
_REPO_PART = re.compile(r"^[A-Za-z0-9_.-]+$")


def github_repo(url: str) -> str | None:
    """"owner/repo" for a github.com URL (https, ssh or scp form), else None. User info is allowed only as `git@`
    in the ssh forms; a port, any other host, or a host that only starts with github.com is refused."""
    url = str(url).strip()
    m = _GH_SCP.match(url)
    if m:
        path = m.group("path")
    else:
        try:
            parts = urlsplit(url)
        except ValueError:
            return None
        if parts.scheme not in ("https", "ssh") or parts.netloc not in ("github.com", "git@github.com"):
            return None
        if parts.scheme == "https" and parts.netloc != "github.com":
            return None
        path = parts.path.lstrip("/")
    path = path.rstrip("/")
    if path.endswith(".git"):
        path = path[:-4]
    bits = path.split("/")
    if len(bits) != 2 or not all(_REPO_PART.match(b) and b not in (".", "..") for b in bits):
        return None
    return "/".join(bits)


def _ref_sha(ls_remote_output: str, ref: str) -> str | None:
    """The sha on the line of `git ls-remote` output whose ref is exactly `ref`. The command also lists refs that only
    end in the pattern (a branch `a/refs/heads/main`), so the first line is not enough."""
    for line in ls_remote_output.splitlines():
        parts = line.split("\t")
        if len(parts) == 2 and parts[1].strip() == ref:
            return parts[0].strip()
    return None


def _commits(n: int) -> str:
    return f"{n} commit" if n == 1 else f"{n} commits"


def _out(runner: ReadOnlyRunner, argv, *, network: bool = False) -> str | None:
    rc, out, _ = runner(argv, network=network)
    return out.strip() if rc == 0 else None


def _count(runner: ReadOnlyRunner, root: str, repo: str, head: str, remote: str) -> tuple[int | None, int | None]:
    """(ahead, behind) of HEAD against upstream main: from the local objects when the remote commit is here, else
    from GitHub's compare API. (None, None) when neither works."""
    if runner(["git", "-C", root, "cat-file", "-e", f"{remote}^{{commit}}"], network=False)[0] == 0:
        text = _out(runner, ["git", "-C", root, "rev-list", "--left-right", "--count", f"{head}...{remote}"])
        try:
            ahead, behind = (int(x) for x in (text or "").split())
            return ahead, behind
        except ValueError:
            pass
    text = _out(runner, ["gh", "api", f"repos/{repo}/compare/{head}...{remote}"], network=True)
    try:
        data = json.loads(text or "")
        behind, ahead = data["ahead_by"], data["behind_by"]      # upstream is ahead_by commits past HEAD
        if all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in (ahead, behind)):
            return ahead, behind
    except (ValueError, KeyError, TypeError):
        pass
    return None, None


def check_clone(path, *, source: str, runner: ReadOnlyRunner, cache: Cache, update: Callable[..., str],
                reuse_s: float = 0.0, branch_warning: bool = True, untracked: bool = True) -> Freshness:
    """Compare a git checkout's HEAD with its origin's main branch. `update(root, branch, remote)` gives the update
    hint; `remote` is upstream main's sha once it is known, else None.
    `untracked` False leaves untracked files out of the dirty-tree warning."""
    path = str(path)
    root = _out(runner, ["git", "-C", path, "rev-parse", "--show-toplevel"])
    result = Freshness(source=source, path=root or path, state=UNKNOWN, detail="")
    key = f"{source}:{root or path}"
    if not root:
        result.detail = f"{path} is not in a git checkout, so it cannot be compared with upstream."
        return result
    result.local = _out(runner, ["git", "-C", root, "rev-parse", "HEAD"])
    result.branch = _out(runner, ["git", "-C", root, "symbolic-ref", "--short", "-q", "HEAD"]) or None
    status = _out(runner, ["git", "-C", root, "status", "--porcelain",
                           "--untracked-files=normal" if untracked else "--untracked-files=no"])
    result.dirty = bool(status) if status is not None else None
    if result.dirty:
        result.warnings.append(f"{root} has uncommitted changes.")
    if branch_warning and result.branch and result.branch != "main":
        result.warnings.append(f"{root} is on branch {result.branch}; it is compared with upstream main.")
    if not result.local:
        result.detail = f"HEAD could not be read in {root}."
        return result
    origin = _out(runner, ["git", "-C", root, "remote", "get-url", "origin"]) or ""
    repo = github_repo(origin)
    if repo is None:
        result.detail = (f"Its origin ({origin or 'none'}) is not a github.com repository, so it is not contacted "
                         "and cannot be compared with upstream.")
        return result
    result.update = update(root, result.branch)
    if _reused(result, cache, key, reuse_s):
        return result
    rc, out, err = runner(["git", "-C", root, "ls-remote", "origin", "refs/heads/main"], network=True)
    remote = _ref_sha(out, "refs/heads/main") if rc == 0 else None
    if not remote or not re.fullmatch(r"[0-9a-f]{40}", remote):
        why = (err.strip().splitlines() or ["no answer"])[-1][:200] if rc != 0 else "origin has no main branch"
        return _unknown(result, why, cache, key)
    result.remote = remote
    if remote == result.local:
        result.state, result.behind, result.ahead = CURRENT, 0, 0
        result.detail = f"HEAD {result.local[:9]} is upstream main ({repo})."
        result.update = None
    else:
        ahead, behind = _count(runner, root, repo, result.local, remote)
        result.ahead, result.behind = ahead, behind
        result.update = update(root, result.branch, remote)
        if behind is None:
            result.state = DIFFERS
            result.detail = (f"HEAD {result.local[:9]} is not upstream main {remote[:9]} ({repo}); how far apart they "
                             "are could not be counted.")
        elif behind > 0:
            result.state = BEHIND
            more = f" and has {_commits(ahead)} of its own" if ahead else ""
            result.detail = (f"HEAD {result.local[:9]} is {_commits(behind)} behind upstream main {remote[:9]} "
                             f"({repo}){more}.")
        else:
            result.state = AHEAD
            result.detail = f"HEAD {result.local[:9]} is {_commits(ahead)} ahead of upstream main {remote[:9]} ({repo})."
            result.update = None
    _remember(result, cache, key)
    return result


def skills_update(root: str, branch: str | None, remote: str | None = None) -> str:
    """How to bring a tenstorrent/skills clone up to date. On a branch other than main a pull would merge main into
    that branch, so the hint says what to do and leaves the choice to the operator."""
    if branch in (None, "main"):
        return f"git -C {root} pull --ff-only origin main"
    return (f"the clone is on branch {branch}. Commit your work there, then `git -C {root} switch main` and update "
            f"main with a fast-forward from origin, or rebase {branch} onto origin/main yourself")


def check_skills_dirs(dirs, *, run=None, cache: Cache | None = None, reuse_s: float = 0.0,
                      offline: bool = False) -> list[Freshness]:
    """One result per git checkout that holds a directory in `dirs` (bringup.toml's skills_dirs). Two directories in
    one clone are checked once. A directory outside any checkout is an UNKNOWN result of its own."""
    runner = run if isinstance(run, ReadOnlyRunner) else ReadOnlyRunner(run or _raw_run, offline=offline)
    cache = cache if cache is not None else Cache()
    out: list[Freshness] = []
    seen: set[str] = set()
    for d in dirs:
        root = _out(runner, ["git", "-C", str(d), "rev-parse", "--show-toplevel"]) if Path(d).is_dir() else None
        if root in seen:
            continue
        if root:
            seen.add(root)
        if not Path(d).is_dir():
            out.append(Freshness(SOURCE_SKILLS, str(d), UNKNOWN, f"{d} does not exist."))
            continue
        out.append(check_clone(d, source=SOURCE_SKILLS, runner=runner, cache=cache, update=skills_update,
                               reuse_s=reuse_s))
    return out


def ttlang_update(root: str, branch: str | None, remote: str | None = None) -> str:
    """How to move to tt-lang's latest main. The checkout is pinned in code (defaults.TTLANG_PIN), so the fix is a
    code change and a rebuild; the override runs on the pinned version."""
    target = f"upstream main {remote}" if remote else "upstream main"
    return (f"The fix is to move the pin: in orchard/defaults.py set TTLANG_PIN to {target} and set "
            "TTLANG_DIAGNOSTICS_SHA256 to the sha256 of python/ttl/diagnostics.py at that commit (review the file if it "
            f"changed), move {root} aside, then run `tt-orchard kernel-env`. To run on the pinned version until then, "
            f"pass {OVERRIDE}")


def check_ttlang_source(src_dir, *, run=None, cache: Cache | None = None, reuse_s: float = 0.0,
                        offline: bool = False) -> Freshness:
    """The tt-lang checkout kernel-env made. Its HEAD is the pin and detached, so no branch warning is given. Untracked
    files do not count as changes: kernel-env's own wheel build leaves packaging/sim/build/ in the checkout."""
    runner = run if isinstance(run, ReadOnlyRunner) else ReadOnlyRunner(run or _raw_run, offline=offline)
    cache = cache if cache is not None else Cache()
    if not Path(src_dir).is_dir():
        return Freshness(SOURCE_TTLANG, str(src_dir), UNKNOWN, f"{src_dir} does not exist; run `tt-orchard kernel-env`.")
    return check_clone(src_dir, source=SOURCE_TTLANG, runner=runner, cache=cache, update=ttlang_update,
                       reuse_s=reuse_s, branch_warning=False, untracked=False)


# ---- the installed tt-lang skill --------------------------------------------------------------------------------

def git_blob_sha(data: bytes) -> str:
    """The sha git gives a file with these bytes (`git hash-object`)."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def upstream_skill_md(data: bytes, upstream_name: str = TTLANG_SKILL_UPSTREAM_NAME) -> bytes:
    """An installed SKILL.md with tt-warp's two edits undone (tt_warp/fastmodels.py, transform_skill_md): the
    frontmatter's `name: ttm-<name>` goes back to `name: <name>`, and the provenance block tt-warp put after the
    frontmatter (a blank line, `> **Provenance:**` and the `> ` lines after it, a blank line) is removed. A file
    without these edits comes back unchanged."""
    lines = data.decode("utf-8", errors="surrogateescape").splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return data
    close = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if close is None:
        return data
    for i in range(1, close):
        if lines[i].rstrip("\n") == f"name: ttm-{upstream_name}":
            lines[i] = f"name: {upstream_name}\n"
            break
    j = close + 1
    if j + 1 < len(lines) and lines[j] == "\n" and lines[j + 1].startswith("> **Provenance:**"):
        k = j + 1
        while k < len(lines) and lines[k].startswith("> "):
            k += 1
        if k < len(lines) and lines[k] == "\n":
            del lines[j:k + 1]
    return "".join(lines).encode("utf-8", errors="surrogateescape")


def ttlang_skill_tree_path() -> str:
    """The `gh api` path of the upstream skill's tree: every file under it, with blob shas, in one answer."""
    return f"repos/{TTLANG_SKILL_REPO}/git/trees/{TTLANG_SKILL_REF}:{TTLANG_SKILL_PATH}?recursive=1"


def _read_regular(path: Path) -> bytes | None:
    """The bytes of a regular file of at most FILE_MAX_BYTES, opened without following a link or blocking on a FIFO;
    None otherwise. The type is checked with lstat before the open, so a device node or a FIFO is never opened, and
    again on the open file."""
    try:
        if not stat.S_ISREG(os.lstat(path).st_mode):
            return None
    except OSError:
        return None
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    except OSError:
        return None
    with os.fdopen(fd, "rb") as f:
        if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
            return None
        data = f.read(FILE_MAX_BYTES + 1)
    return data if len(data) <= FILE_MAX_BYTES else None


def local_skill_shas(skill_dir: Path, preread: dict[str, bytes] | None = None) -> dict[str, str | None]:
    """{relative path: git blob sha} of every file under the skill directory, SKILL.md after upstream_skill_md. A
    file that is not a regular file (a link, a FIFO) or is too large maps to None. Directories that are links are
    not entered. `preread` maps relative paths to bytes already read (the probe's reference copies); those bytes are
    hashed and the files are not read again, so the check and the copy cannot see different contents."""
    preread = preread or {}
    out: dict[str, str | None] = {}
    for dirpath, dirnames, filenames in os.walk(skill_dir, followlinks=False):
        for name in filenames + [d for d in dirnames if os.path.islink(os.path.join(dirpath, d))]:
            p = Path(dirpath) / name
            rel = p.relative_to(skill_dir).as_posix()
            data = preread[rel] if rel in preread else _read_regular(p)
            if data is not None and rel == "SKILL.md":
                data = upstream_skill_md(data)
            out[rel] = git_blob_sha(data) if data is not None else None
    return out


_VERSION_ROW = re.compile(r"^\|\s*(\d+\.\d+)\s*\|")
# "## Specification Versions" in tt-lang's own copy, "## 0. Versions" in the older copy the skill carries.
_VERSIONS_HEADING = re.compile(r"^## (?:\d+\. )?(?:Specification )?Versions[ \t]*$", re.MULTILINE)


def spec_version(data: bytes) -> str | None:
    """The last version in a TTLangSpecification.md's versions table ("## Specification Versions" or
    "## 0. Versions"), or None."""
    text = data.decode("utf-8", errors="replace")
    head = _VERSIONS_HEADING.search(text)
    if head is None:
        return None
    rest = text[head.end():].split("\n## ", 1)[0]
    found = [m.group(1) for m in (_VERSION_ROW.match(line) for line in rest.splitlines()) if m]
    return found[-1] if found else None


def _version_key(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in v.split("."))


def spec_versions(skill_dir: Path, src_dir: Path | None) -> tuple[str | None, str | None]:
    """(the skill's spec version, the tt-lang checkout's spec version)."""
    mine = _read_regular(Path(skill_dir) / "TTLangSpecification.md")
    theirs = _read_regular(Path(src_dir) / "docs" / "sphinx" / "specs" / "TTLangSpecification.md") if src_dir else None
    return (spec_version(mine) if mine else None), (spec_version(theirs) if theirs else None)


def check_ttlang_skill(skill_dir, *, run=None, cache: Cache | None = None, src_dir=None, reuse_s: float = 0.0,
                       offline: bool = False, preread: dict[str, bytes] | None = None) -> Freshness:
    """Compare every file of the installed tt-lang skill with the upstream branch's tree (one `gh api` call).

    CURRENT when every upstream file is here with the same blob sha (SKILL.md after the installer's edits are undone);
    DIFFERS otherwise, naming the files. Local files upstream does not have are listed as a warning. With `src_dir`,
    a skill that documents an older tt-lang specification than the checkout gets a warning: the skill is the latest
    upstream has, and it is still older than the tt-lang the simulator runs."""
    runner = run if isinstance(run, ReadOnlyRunner) else ReadOnlyRunner(run or _raw_run, offline=offline)
    cache = cache if cache is not None else Cache()
    skill_dir = Path(skill_dir)
    upstream = f"{TTLANG_SKILL_REPO} branch {TTLANG_SKILL_REF}, {TTLANG_SKILL_PATH}"
    result = Freshness(SOURCE_TTLANG_SKILL, str(skill_dir), UNKNOWN, "", update=TTLANG_SKILL_UPDATE)
    if not skill_dir.is_dir():
        result.detail = (f"{skill_dir} does not exist. Install the tt-lang skill (`tt-warp fastmodels sync`) or name "
                         "another directory with --tt-lang-skill-dir or ORCHARD_TTLANG_SKILL.")
        return result
    local = local_skill_shas(skill_dir, preread)
    result.local = hashlib.sha256(json.dumps(local, sort_keys=True).encode()).hexdigest()
    if src_dir is not None:
        mine, theirs = spec_versions(skill_dir, Path(src_dir))
        if mine and theirs and _version_key(mine) < _version_key(theirs):
            result.warnings.append(
                f"the skill documents tt-lang specification {mine}; the tt-lang checkout documents {theirs}. Where "
                "they differ (for example `ttl.kernel` became `ttl.operation` in 0.13), the checkout's specification "
                "and examples are the ones the simulator runs.")
    key = f"{SOURCE_TTLANG_SKILL}:{skill_dir}"
    if _reused(result, cache, key, reuse_s):
        return result
    rc, out, err = runner(["gh", "api", ttlang_skill_tree_path()], network=True)
    if rc != 0:
        return _unknown(result, (err.strip().splitlines() or ["no answer"])[-1][:200], cache, key)
    try:
        data = json.loads(out)
        if data.get("truncated") is not False or not isinstance(data.get("tree"), list):
            raise ValueError("a truncated or malformed tree")
        remote = {e["path"]: e["sha"] for e in data["tree"] if e.get("type") == "blob"}
        if not remote or not all(isinstance(k, str) and isinstance(v, str) for k, v in remote.items()):
            raise ValueError("an empty tree")
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        return _unknown(result, f"GitHub's answer was not usable: {exc}", cache, key)
    result.remote = hashlib.sha256(json.dumps(remote, sort_keys=True).encode()).hexdigest()
    changed = sorted(p for p, sha in remote.items() if local.get(p) != sha)
    extra = sorted(p for p in local if p not in remote)
    if extra:
        result.warnings.append(f"files upstream does not have: {', '.join(extra)}.")
    if changed:
        result.state = DIFFERS
        result.detail = f"These files differ from {upstream} or are missing here: {', '.join(changed)}."
    else:
        result.state = CURRENT
        result.detail = f"Every file matches {upstream}."
        result.update = None
    _remember(result, cache, key)
    return result


def check_kernel_sources(src_dir, skill_dir, *, run=None, cache: Cache | None = None, offline: bool = False,
                         preread: dict[str, bytes] | None = None) -> list[Freshness]:
    """The two sources the kernel probe gives the coder: the tt-lang checkout and the installed tt-lang skill. One
    runner (and so one network budget) and one cache serve both."""
    runner = run if isinstance(run, ReadOnlyRunner) else ReadOnlyRunner(run or _raw_run, offline=offline)
    cache = cache if cache is not None else Cache()
    return [check_ttlang_source(src_dir, run=runner, cache=cache),
            check_ttlang_skill(skill_dir, run=runner, cache=cache, src_dir=src_dir, preread=preread)]
