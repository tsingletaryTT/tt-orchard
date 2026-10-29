# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`tt-orchard verify-package REPO_ID`: pull a published package from the Hub into a scratch directory and check it (issue 24).

A package is checked before it is uploaded. This command checks what readers actually get: the repository as the Hub serves it.

What it does, in order:

1. Reads the repository's commit and file list from the Hub (`Hub.info`). `--plan` stops here and prints what a pull would fetch.
2. Pulls every file of that commit into the scratch directory (`Hub.download`). A private repository is pulled with the token in
   the `HF_TOKEN` environment variable. The token is never printed and never written to a file.
3. Runs the checks below on the pulled files and prints one line each (ok, warn or FAIL):
   - the pulled bytes match the Hub's own sha256 for every file the Hub records one for (the large files);
   - no weights, caches, symbolic links, tokens, home paths or host names (the same scrub the staging step runs, orchard/scrub.py);
   - a license file or a `license:` line in the card's front matter;
   - the package's own sha256 record (`package-record.json` or `SHA256SUMS`): every recorded file is present with the recorded
     hash, and no file is missing from the record;
   - JSON manifests parse; `tt-model.yaml`, when present, names this repository and a 40-character weights revision;
   - the revision the card names matches the revision the package record, `tt-model.yaml` and the launchers name, and the weights
     repository has that revision on the Hub;
   - with `--staged DIR`, the pulled file set and hashes equal that local staged directory.
4. Prints the command that boots the package, as text.

It never opens a device, takes a lease, resets a chip, uploads or publishes. The boot is not run here. The printed command goes to
an agent that holds a hardware lease; it is `gozer run ... -- bash run.sh` with the chip count the package's manifest or card names.

The Hub is injected: `Hub` is the interface, `HubClient` the real one (huggingface_hub), and tests pass a fake that serves a
directory, so no test needs the network.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from orchard.scrub import scrub_package

OK, WARN, FAIL = "ok", "warn", "FAIL"
EXIT_OK, EXIT_FAILED, EXIT_REFUSED = 0, 1, 2
HEX40 = re.compile(r"^[0-9a-f]{40}$")
REVISION_NEAR = re.compile(r"revision\W{0,8}([0-9a-f]{40})\b", re.IGNORECASE)
IGNORED = frozenset({".gitattributes"})          # the Hub adds this file; no package record lists it
RECORD_FILES = ("package-record.json", "SHA256SUMS")
LICENSE_FILES = re.compile(r"^(LICENSE|LICENCE|COPYING)([-._].*)?$", re.IGNORECASE)
WHO = "claude:verify-package"


@dataclass(frozen=True)
class RepoFile:
    path: str
    size: int
    sha256: str | None = None        # the Hub's sha256, which it records for large (LFS) files only


@dataclass
class RepoInfo:
    sha: str
    private: bool
    files: list[RepoFile] = field(default_factory=list)


class Hub:
    """What the command needs from the Hub. Tests subclass this; `HubClient` is the real one."""

    def info(self, repo: str, revision: str | None = None) -> RepoInfo:
        raise NotImplementedError

    def download(self, repo: str, revision: str, dest: Path, paths: list[str]) -> None:
        raise NotImplementedError

    def revision_exists(self, repo: str, revision: str) -> bool | None:
        """True or False when the Hub answers, None when it cannot say (gated, private to this token, offline)."""
        raise NotImplementedError


class HubClient(Hub):
    """The real Hub through huggingface_hub. The token comes from `HF_TOKEN` and goes only to the library."""

    def __init__(self, token: str | None = None):
        self.token = token if token is not None else (os.environ.get("HF_TOKEN") or None)

    def _api(self):
        from huggingface_hub import HfApi
        return HfApi(token=self.token)

    def info(self, repo, revision=None):
        m = self._api().model_info(repo, revision=revision, files_metadata=True)
        files = [RepoFile(s.rfilename, s.size or 0, (s.lfs.sha256 if getattr(s, "lfs", None) else None))
                 for s in (m.siblings or [])]
        return RepoInfo(sha=m.sha, private=bool(m.private), files=files)

    def download(self, repo, revision, dest, paths):
        from huggingface_hub import hf_hub_download
        for p in paths:
            hf_hub_download(repo, p, revision=revision, local_dir=str(dest), token=self.token)

    def revision_exists(self, repo, revision):
        try:
            self._api().model_info(repo, revision=revision)
            return True
        except Exception as exc:                  # not found is an answer; anything else is no answer
            name = type(exc).__name__
            return False if name in ("RevisionNotFoundError", "EntryNotFoundError") else None


@dataclass
class Finding:
    name: str
    status: str
    detail: str


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def hash_tree(root: Path) -> dict[str, str]:
    """{relative path: sha256} for every regular file under root, except the ones the Hub adds."""
    out = {}
    for dirpath, _, names in os.walk(root):
        for n in names:
            f = Path(dirpath) / n
            rel = f.relative_to(root).as_posix()
            if rel in IGNORED or rel.startswith(".cache/") or f.is_symlink():
                continue
            out[rel] = sha256_of(f)
    return out


def _text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _front_matter(card: str | None) -> dict[str, str]:
    """The top-level `key: value` lines of the card's YAML front matter (lists and nested values are skipped)."""
    if not card or not card.startswith("---"):
        return {}
    end = card.find("\n---", 3)
    out = {}
    for line in (card[3:end] if end > 0 else "").splitlines():
        m = re.match(r"^([A-Za-z_][\w-]*):\s*(\S.*)$", line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def _tags(card: str | None) -> list[str]:
    if not card or not card.startswith("---"):
        return []
    end = card.find("\n---", 3)
    block = card[3:end] if end > 0 else ""
    m = re.search(r"^tags:\s*\n((?:\s*-\s*\S.*\n?)+)", block, re.MULTILINE)
    return [t.strip() for t in re.findall(r"^\s*-\s*(\S.*)$", m.group(1), re.MULTILINE)] if m else []


# ---- the checks --------------------------------------------------------------------------------------------

def check_hub_hashes(root: Path, info: RepoInfo) -> Finding:
    bad, checked = [], 0
    for f in info.files:
        if f.sha256 is None:
            continue
        p = root / f.path
        checked += 1
        if not p.is_file():
            bad.append(f"{f.path} was not pulled")
        elif sha256_of(p) != f.sha256:
            bad.append(f"{f.path} differs from the Hub's sha256")
    if bad:
        return Finding("hub-hashes", FAIL, "; ".join(bad))
    return Finding("hub-hashes", OK, f"{checked} large files match the Hub's sha256" if checked
                   else "the Hub records no sha256 for these files (none are large), so only the package's own record applies")


def check_scrub(root: Path, hostname: str | None, home: str | None) -> Finding:
    hits = scrub_package(root, hostname=hostname, home=home)
    if hits:
        return Finding("scrub", FAIL, f"{len(hits)} finding{'s' if len(hits) != 1 else ''}: " + "; ".join(hits[:12])
                       + (" ..." if len(hits) > 12 else ""))
    return Finding("scrub", OK, "no weights, caches, symbolic links, tokens, home paths or host names")


def check_license(root: Path, card: str | None) -> Finding:
    files = sorted(p.name for p in root.iterdir() if p.is_file() and LICENSE_FILES.match(p.name))
    declared = _front_matter(card).get("license")
    if files or declared:
        bits = ([", ".join(files)] if files else []) + ([f"card license: {declared}"] if declared else [])
        return Finding("license", OK, "; ".join(bits))
    return Finding("license", WARN, "no license file and no `license:` line in the card's front matter")


def _read_record(root: Path) -> tuple[str, dict[str, str], str | None] | None:
    """(record file name, {path: sha256}, the record's revision) from the first record the package ships, else None."""
    rec = root / "package-record.json"
    if rec.is_file():
        try:
            doc = json.loads(rec.read_text(encoding="utf-8"))
            listed = {e["path"]: e["sha256"] for e in doc.get("files", []) if isinstance(e, dict)
                      and isinstance(e.get("path"), str) and isinstance(e.get("sha256"), str)}
            rev = doc.get("revision")
            return "package-record.json", listed, rev if isinstance(rev, str) else None
        except (ValueError, AttributeError) as exc:
            return "package-record.json", {"": f"unreadable: {exc}"}, None
    sums = root / "SHA256SUMS"
    if sums.is_file():
        listed = {}
        for line in sums.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.match(r"^([0-9a-fA-F]{64})\s+\*?(\S.*)$", line)
            if m:
                listed[m.group(2)] = m.group(1).lower()
        return "SHA256SUMS", listed, None
    return None


def check_record(root: Path, record) -> Finding:
    if record is None:
        return Finding("record", WARN, "the package ships no package-record.json or SHA256SUMS, so only the Hub's hashes apply")
    name, listed, _ = record
    if "" in listed:
        return Finding("record", FAIL, f"{name} {listed['']}")
    problems = []
    for path, want in sorted(listed.items()):
        p = root / path
        if not p.is_file():
            problems.append(f"{path} is recorded and absent")
        elif sha256_of(p) != want:
            problems.append(f"{path} differs from its recorded sha256")
    unrecorded = sorted(set(hash_tree(root)) - set(listed) - set(RECORD_FILES))
    if unrecorded:
        problems.append("not in the record: " + ", ".join(unrecorded[:8]) + (" ..." if len(unrecorded) > 8 else ""))
    if problems:
        return Finding("record", FAIL, f"{name}: " + "; ".join(problems))
    return Finding("record", OK, f"{name}: all {len(listed)} recorded files are present and match")


def _yaml_scalar(text: str, pattern: str) -> str | None:
    m = re.search(pattern, text, re.MULTILINE)
    return m.group(1).strip().strip("'\"") if m else None


def check_manifests(root: Path, repo: str) -> Finding:
    problems, seen = [], []
    for p in sorted(root.rglob("*.json")):
        rel = p.relative_to(root).as_posix()
        if rel.startswith(".cache/") or "manifest" not in p.name and p.name != "vllm_metadata.json":
            continue
        seen.append(rel)
        try:
            json.loads(p.read_text(encoding="utf-8"))
        except ValueError as exc:
            problems.append(f"{rel} is not JSON ({exc})")
    y = root / "tt-model.yaml"
    if y.is_file():
        seen.append("tt-model.yaml")
        text = y.read_text(encoding="utf-8", errors="replace")
        named = _yaml_scalar(text, r"^repo:\s*(\S+)")
        if named is not None and named != repo:
            problems.append(f"tt-model.yaml names repo {named}; the package is {repo}")
        rev = _yaml_scalar(text, r"^  revision:\s*(\S+)")
        if rev is not None and not HEX40.match(rev):
            problems.append(f"tt-model.yaml's weights revision {rev!r} is not a 40-character commit")
    if problems:
        return Finding("manifests", FAIL, "; ".join(problems))
    return Finding("manifests", OK, ("checked " + ", ".join(seen)) if seen else "the package ships no manifest to check")


def card_revisions(card: str | None) -> set[str]:
    return {m.lower() for m in REVISION_NEAR.findall(card or "")}


def check_revision(root: Path, card: str | None, record, hub: Hub | None) -> Finding:
    named = card_revisions(card)
    if not named:
        return Finding("revision", WARN, "the card names no pinned weights revision (a 40-character commit after the word revision)")
    problems, notes = [], []
    rrev = record[2] if record else None
    if rrev and rrev not in named:
        problems.append(f"package record revision {rrev} is not the card's ({', '.join(sorted(named))})")
    for fname in ("tt-model.yaml", "run.sh", "install.sh"):
        t = _text(root / fname)
        for rev in sorted({m.lower() for m in REVISION_NEAR.findall(t or "")} - named):
            notes.append(f"{fname} names revision {rev}, which the card does not")
    model = None
    rec = root / "package-record.json"
    if rec.is_file():
        try:
            model = json.loads(rec.read_text(encoding="utf-8")).get("model")
        except (ValueError, AttributeError):
            model = None
    model = model if isinstance(model, str) else _front_matter(card).get("base_model")
    if hub is not None and model and "/" in model:
        for rev in sorted(named):
            found = hub.revision_exists(model, rev)
            if found is False:
                problems.append(f"{model} has no revision {rev} on the Hub")
            elif found is None:
                notes.append(f"could not ask the Hub whether {model} has revision {rev}")
    if problems:
        return Finding("revision", FAIL, "; ".join(problems + notes))
    detail = f"the card pins {', '.join(sorted(named))}" + (f" ({model})" if model else "")
    return Finding("revision", WARN if notes else OK, detail + ("; " + "; ".join(notes) if notes else ""))


def compare_staged(pulled: Path, staged: Path) -> Finding:
    a, b = hash_tree(pulled), hash_tree(staged)
    only_hub = sorted(set(a) - set(b))
    only_staged = sorted(set(b) - set(a))
    differ = sorted(p for p in set(a) & set(b) if a[p] != b[p])
    if not (only_hub or only_staged or differ):
        return Finding("staged", OK, f"the pulled files and the staged directory are identical ({len(a)} files)")
    parts = []
    for label, names in (("only on the Hub", only_hub), ("only in the staged directory", only_staged), ("different bytes", differ)):
        if names:
            parts.append(f"{label}: " + ", ".join(names[:8]) + (" ..." if len(names) > 8 else ""))
    return Finding("staged", FAIL, "; ".join(parts))


def chips_for(root: Path, card: str | None) -> tuple[int, str]:
    """The chip count the package names (tt-model.yaml's mesh_device, else a card tag such as p150x2) and where it came from."""
    y = _text(root / "tt-model.yaml") or ""
    for source, text in (("tt-model.yaml mesh_device", _yaml_scalar(y, r"^\s*mesh_device:\s*(\S+)")),
                         *(("card tag", t) for t in _tags(card))):
        m = re.fullmatch(r"[Pp](\d+)(?:x(\d+))?", text or "")
        if m:
            boards = int(m.group(2) or 1)
            return boards * (2 if m.group(1) == "300" else 1), f"{source} {text}"
    return 1, "no mesh named in the package; one chip is the smallest it can need"


def boot_text(root: Path, repo: str, sha: str, card: str | None) -> list[str]:
    chips, why = chips_for(root, card)
    lines = [f"Boot command for {repo} at {sha[:12]} (run it by a hardware agent; verify-package never opens a device).",
             f"The package names {chips} chip{'s' if chips != 1 else ''} ({why}).", ""]
    if not (root / "run.sh").is_file():
        return lines + ["The package ships no run.sh, so there is no boot command to print. Read its card."]
    lines.append(f"cd {root}")
    if (root / "install.sh").is_file():
        lines.append("bash install.sh        # builds the package's venv; opens no device")
    lines.append(f'gozer run --chips {chips} --who "{WHO}" --reason "boot check of {repo}@{sha[:12]}" -- bash run.sh')
    lines += ["", "Pass `HF_TOKEN` through if the base weights repository is gated. Stop the server and release the lease when the "
              "check is done. `gozer run` resets the chips it leased on exit."]
    return lines


# ---- the whole command -------------------------------------------------------------------------------------

def verify(repo: str, scratch: Path, hub: Hub, *, staged: Path | None = None, revision: str | None = None,
           hostname: str | None = None, home: str | None = None, say=print) -> tuple[list[Finding], list[str]]:
    """Pull and check. Returns the findings and the boot text."""
    info = hub.info(repo, revision)
    say(f"{repo}: commit {info.sha}, {'private' if info.private else 'public'}, {len(info.files)} files, "
        f"{sum(f.size for f in info.files)} bytes")
    hub.download(repo, info.sha, scratch, [f.path for f in info.files])
    shutil.rmtree(scratch / ".cache", ignore_errors=True)      # huggingface_hub writes this directory itself; it holds no package content
    card = _text(scratch / "README.md")
    record = _read_record(scratch)
    findings = [check_hub_hashes(scratch, info), check_scrub(scratch, hostname, home), check_license(scratch, card),
                check_record(scratch, record), check_manifests(scratch, repo), check_revision(scratch, card, record, hub)]
    if staged is not None:
        findings.append(compare_staged(scratch, staged))
    return findings, boot_text(scratch, repo, info.sha, card)


def render_plan(repo: str, info: RepoInfo, scratch: Path) -> list[str]:
    big = sorted(info.files, key=lambda f: -f.size)[:5]
    lines = [f"plan for {repo}: commit {info.sha}, {'private' if info.private else 'public'}, {len(info.files)} files, "
             f"{sum(f.size for f in info.files)} bytes, into {scratch}", "largest files:"]
    lines += [f"  {f.size:>12}  {f.path}" for f in big]
    lines.append("nothing was pulled (--plan)")
    return lines


def main(args, *, out=None, hub: Hub | None = None, hostname=None, home=None) -> int:
    """Exit 0 when every check passes or warns (or without --check), 1 when --check is given and a check fails, 2 on a refusal."""
    out = out or sys.stdout
    token = os.environ.get("HF_TOKEN") or ""

    def say(text=""):
        if token and token in text:                # the token is never printed, whatever a library put in a message
            text = text.replace(token, "<token>")
        print(text, file=out, flush=True)
    repo = args.repo_id
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        print(f"refused: {repo!r} is not a Hub repository id (ORG/NAME)", file=sys.stderr)
        return EXIT_REFUSED
    staged = Path(args.staged) if getattr(args, "staged", None) else None
    if staged is not None and not staged.is_dir():
        print(f"refused: --staged {staged} is not a directory", file=sys.stderr)
        return EXIT_REFUSED
    hub = hub or HubClient()
    scratch = Path(args.scratch) if getattr(args, "scratch", None) else None
    if scratch is not None and scratch.exists() and not scratch.is_dir():
        print(f"refused: --scratch {scratch} exists and is not a directory", file=sys.stderr)
        return EXIT_REFUSED
    if scratch is not None and scratch.exists() and any(scratch.iterdir()):
        print(f"refused: --scratch {scratch} is not empty; the pull would mix files", file=sys.stderr)
        return EXIT_REFUSED
    try:
        if getattr(args, "plan", False):
            info = hub.info(repo, getattr(args, "revision", None))
            for line in render_plan(repo, info, scratch or Path("<a new temporary directory>")):
                say(line)
            return EXIT_OK
        if scratch is None:
            scratch = Path(tempfile.mkdtemp(prefix="orchard-verify-"))
        scratch.mkdir(parents=True, exist_ok=True)
        findings, boot = verify(repo, scratch, hub, staged=staged, revision=getattr(args, "revision", None),
                                hostname=hostname, home=home, say=say)
    except Exception as exc:                      # a Hub or network failure is a refusal to judge and never counts as a pass
        print(f"refused: could not read {repo} from the Hub: {type(exc).__name__}: {(str(exc).replace(token, '<token>') if token else str(exc))[:200]}",
              file=sys.stderr)
        return EXIT_REFUSED
    say(f"pulled into {scratch}")
    for f in findings:
        say(f"  {f.status:<4} {f.name}: {f.detail}")
    failed = [f for f in findings if f.status == FAIL]
    say("")
    say(f"{len(findings) - len(failed)} of {len(findings)} checks passed" + (f"; {len(failed)} failed" if failed else ""))
    say("")
    for line in boot:
        say(line)
    return EXIT_FAILED if failed and getattr(args, "check", False) else EXIT_OK
