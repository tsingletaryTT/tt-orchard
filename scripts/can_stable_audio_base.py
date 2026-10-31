#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Assemble the base directory that `config/can.stable-audio-3.toml` is staged from.

    python3 scripts/can_stable_audio_base.py --out BASE --library ~/src/stable-audio-3 --snapshot <HF snapshot directory of the pinned revision> \
        [--evidence DIR] [--listening-note "2026-10-29: ..."]
    tt-orchard can stage config/can.stable-audio-3.toml --base BASE --out OUT/stable-audio-3-small-music-blackhole

Why a script: the can's spec names files inside one base directory. Four things the bundle needs are not in this repository, so the script copies
them from where they are, after checking that they are the ones the card names:

  * the model library (`stable_audio_3/`, MIT) from a checkout. The checkout's commit must be LIBRARY_COMMIT, and it must have no uncommitted change
    to the package directory, so the bundle holds exactly that commit's code;
  * the model's license files (`LICENSE.md`, `NOTICE`, `LICENSE_GEMMA.md`) from the snapshot of the pinned revision in the Hugging Face cache. The
    snapshot's directory name must be the pinned revision. The license text is the model's, is not in this repository, and the card quotes it word
    for word from this copy;
  * the chip runtime files of this repository (the same list the recipe package copies), with a stand-in `orchard/__init__.py`;
  * the job server, the ASGI front end, the client tool and the card text of this repository.

The card is rendered here: `pkgsrc/can/stable-audio-3-README.md` has placeholders for the license sentences (cut from the copy of `LICENSE.md` by the
same markers the recipe package's builder uses), the NOTICE text and the numbers table. A card that has an unresolved placeholder, a quotation that is
not in the license copy, no "Powered by Stability AI" or no required section is refused.

Nothing here uploads, publishes or opens a device.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from orchard import audio_package as ap  # noqa: E402

MODEL_REVISION = "8bdfb53c6b0fbc62113cd571fd37373efe15fa14"
LIBRARY_COMMIT = "77e0398c26f7d22d801ec1bb036307e8b231f741"
RUNTIME_SRC = (("orchard/resident.py", "orchard/resident.py"), ("orchard/audio_reference.py", "orchard/audio_reference.py"),
               ("orchard/pinned_files.py", "orchard/pinned_files.py"),
               ("orchard/skills/audio-reference-templates/make_reference.py", "orchard/skills/audio-reference-templates/make_reference.py"),
               ("orchard/skills/audio-chip-templates/audio_runtimes.py", "orchard/skills/audio-chip-templates/audio_runtimes.py"),
               ("orchard/skills/audio-chip-templates/chip_dit.py", "orchard/skills/audio-chip-templates/chip_dit.py"),
               ("orchard/skills/audio-chip-templates/chip_decoder.py", "orchard/skills/audio-chip-templates/chip_decoder.py"),
               ("orchard/skills/audio-chip-templates/check_revision.py", "orchard/skills/audio-chip-templates/check_revision.py"))
TOOLS = ("audio_client.py", "cpu_floor.py")
EVIDENCE_FILES = ('boot-check.json', 'clip-comparison.json', 'cpu-floor.json', 'session-recipe-package.json', 'session-s1.json', 'session-s2.json', 'session-s3.json', 'session-ttmodel.json')
CARD_TEMPLATE = "pkgsrc/can/stable-audio-3-README.md"
# The agreement's own paragraphs the card quotes: (placeholder, first words, last words). Cut from the license copy with ap.quote().
EXTRA_QUOTES = (("distribution", "If You distribute or make available the Stability AI Materials", "on a related website, user interface, blogpost, about page, or product documentation."),
                ("restrictions", "Your use of the Stability AI Materials and Derivative Works, including any output", "(excluding the Models or Derivative Works)."))
PLACEHOLDER = re.compile(r"\{\{([A-Z_:a-z0-9]+)\}\}")
REQUIRED_TAGS = ("tt-model-cache", "blackhole", "p150", "thin", "tt-dit-server")
REQUIRED_SECTIONS = ("## Expected performance", "## Limitations", "## Intended use", "## Quickstart", "## Licensing", "## Provenance")


class BaseError(Exception):
    """Something the base directory cannot be built from. The message says what and where."""


def check_library(src: Path, commit: str = LIBRARY_COMMIT) -> Path:
    """The `stable_audio_3` package directory of a checkout whose HEAD is `commit` and whose package directory has no uncommitted change."""
    pkg = Path(src) / "stable_audio_3"
    if not pkg.is_dir():
        raise BaseError(f"{src} has no stable_audio_3/ directory")
    head = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True)
    if head.returncode != 0 or head.stdout.strip() != commit:
        raise BaseError(f"the library checkout {src} is at {head.stdout.strip() or 'an unknown commit'}, and the card names {commit}")
    dirty = subprocess.run(["git", "-C", str(src), "status", "--porcelain", "--", "stable_audio_3"], capture_output=True, text=True)
    if dirty.returncode != 0 or dirty.stdout.strip():
        raise BaseError(f"the library checkout has uncommitted changes under stable_audio_3/: {dirty.stdout.strip()[:200]}")
    return pkg


def check_snapshot(snapshot: Path, revision: str = MODEL_REVISION) -> Path:
    snap = Path(snapshot)
    if snap.name != revision:
        raise BaseError(f"the snapshot directory is named {snap.name}, and the pinned revision is {revision}")
    return snap


def copy_tree(src: Path, dst: Path) -> int:
    """Copy regular files of `src` under `dst`, skipping caches. A link is refused. Returns the file count."""
    n = 0
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src)
        if "__pycache__" in rel.parts or p.suffix == ".pyc":
            continue
        if p.is_symlink():
            raise BaseError(f"{p} is a link; the base is made of regular files")
        if p.is_file():
            (dst / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dst / rel)
            n += 1
    return n


def quotes(license_text: str) -> dict:
    """Every sentence the card quotes, cut from the license copy. A missing one is an error (the license is not the one this card was written for)."""
    out = {}
    for name, start, end in (*ap.QUOTES, *EXTRA_QUOTES):
        q = ap.quote(license_text, start, end)
        if q is None:
            raise BaseError(f"the license copy has no {name} paragraph ({start[:40]}...); read the agreement again before the card quotes it")
        out[name] = q
    return out


def render_card(template: str, license_bytes: bytes, numbers: str, listening_note: str | None = None) -> str:
    """Fill the placeholders: {{QUOTE:name}} (a license paragraph, word for word), {{ATTRIBUTION}} (the sentence the agreement asks for), {{NUMBERS}},
    {{LISTENING_NOTE}} (the operator's dated judgment, or nothing)."""
    q = quotes(license_bytes.decode("utf-8"))
    attribution = ap.attribution_sentence(license_bytes)
    if attribution is None:
        raise BaseError("the license copy has no attribution sentence")
    note = ap.check_note(listening_note) if listening_note else None

    def fill(m: re.Match) -> str:
        key = m.group(1)
        if key.startswith("QUOTE:"):
            if key[6:] not in q:
                raise BaseError(f"the card template asks for the license paragraph {key[6:]!r}, which is not one of {sorted(q)}")
            return q[key[6:]]
        if key == "ATTRIBUTION":
            return attribution
        if key == "NUMBERS":
            return numbers.strip()
        if key == "LISTENING_NOTE":
            return f"{ap.LISTENING_HEAD}\n\n{note}\n\nThis is the operator's listening judgment on the date above.\n" if note else ""
        raise BaseError(f"the card template has an unknown placeholder {{{{{key}}}}}")
    return PLACEHOLDER.sub(fill, template)


def card_problems(card: str, license_bytes: bytes) -> list[str]:
    """What is wrong with a rendered card. Empty is a pass."""
    problems = []
    if PLACEHOLDER.search(card):
        problems.append("the card has an unresolved placeholder: " + PLACEHOLDER.search(card).group(0))
    m = re.match(r"---\n(.*?)\n---\n", card, re.S)
    if not m:
        problems.append("the card has no front matter")
    else:
        front = m.group(1)
        for tag in REQUIRED_TAGS:
            if not re.search(rf"^- {re.escape(tag)}$", front, re.M):
                problems.append(f"the front matter has no tag {tag!r}")
        if "tt-model-catalog" in front:
            problems.append("the front matter has the tag tt-model-catalog, which `tt-model publish` adds and an author does not")
        for key in ("license: other", "pipeline_tag: text-to-audio", f"base_model:\n- {ap.MODEL_ID}"):
            if key not in front:
                problems.append(f"the front matter lacks {key!r}")
    for heading in REQUIRED_SECTIONS:
        if not re.search(rf"^{re.escape(heading)}\s*$", card, re.M):
            problems.append(f"the card has no {heading!r} section")
    flat = ap._flat(license_bytes.decode("utf-8"))
    after = card.split("\n## Licensing", 1)[-1].split("\n## ", 1)[0]        # the quotations of the agreement are in the Licensing section
    for line in after.splitlines():
        if line.startswith("> ") and line[2:].strip() and ap._flat(line[2:]) not in flat:
            problems.append(f"a quoted line in Licensing is not in the license copy: {line[:80]}")
    for cited in sorted(set(re.findall(r"evidence/[A-Za-z0-9_.-]+\.json", card))):
        if cited.split("/", 1)[1] not in EVIDENCE_FILES:
            problems.append(f"the card cites {cited}, which is not one of the evidence files the bundle ships")
    if ap.POWERED_BY not in card:
        problems.append(f"the card does not display {ap.POWERED_BY!r}, which the agreement asks for")
    return problems


def assemble(out: Path, *, library: Path, snapshot: Path, numbers: str, evidence: Path | None = None, listening_note: str | None = None,
             repo: Path = ROOT, library_commit: str = LIBRARY_COMMIT, revision: str = MODEL_REVISION) -> list[str]:
    """Build `out` (it must not exist). Returns the files made."""
    out = Path(out)
    if out.exists():
        raise BaseError(f"{out} already exists; the base is built in a new directory")
    lib = check_library(library, library_commit)
    snap = check_snapshot(snapshot, revision)
    lic = (snap / "LICENSE.md").resolve().read_bytes()          # a cache snapshot holds links to blobs
    notice_src = (snap / "NOTICE").resolve().read_bytes().decode("utf-8")
    gemma = (snap / "LICENSE_GEMMA.md").resolve().read_bytes()
    template = (repo / CARD_TEMPLATE).read_text(encoding="utf-8")
    card = render_card(template, lic, numbers, listening_note)
    problems = card_problems(card, lic)
    if problems:
        raise BaseError("the card has problems:\n  " + "\n  ".join(problems))
    out.mkdir(parents=True)
    try:
        (out / "README.md").write_text(card, encoding="utf-8")
        shutil.copyfile(repo / "pkgsrc/can/stable_audio_model.py", out / "stable_audio_model.py")
        for name in ("stable_audio_asgi.py", "stable_audio_jobs.py"):
            (out / "extra").mkdir(exist_ok=True)
            shutil.copyfile(repo / "pkgsrc/server" / name, out / "extra" / name)
        for rel, dst in RUNTIME_SRC:
            (out / "runtime" / dst).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(repo / rel, out / "runtime" / dst)
        (out / "runtime/orchard/__init__.py").write_text(ap.RUNTIME_INIT, encoding="utf-8")
        copy_tree(lib, out / "lib" / "stable_audio_3")
        (out / "LICENSE.md").write_bytes(lic)
        (out / "LICENSE_GEMMA.md").write_bytes(gemma)
        (out / "NOTICE").write_text(ap.notice_text(lic) + "\n" + notice_src.strip() + "\n", encoding="utf-8")
        shutil.copyfile(repo / "LICENSE", out / "LICENSE-runtime")
        shutil.copyfile(lib.parent / "LICENSE", out / "LICENSE-stable-audio-3-library")
        (out / "tools").mkdir()
        for tool in TOOLS:
            shutil.copyfile(repo / "pkgsrc/tools" / tool, out / "tools" / tool)
        (out / "evidence").mkdir()
        for name in EVIDENCE_FILES:
            if evidence is None or not (Path(evidence) / name).is_file():
                raise BaseError(f"the evidence directory has no {name}; the spec names it and the card's numbers cite it")
            shutil.copyfile(Path(evidence) / name, out / "evidence" / name)
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)
        raise
    return sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())


def main(argv=None) -> int:
    ap_ = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap_.add_argument("--out", required=True)
    ap_.add_argument("--library", required=True, help="a checkout of the stable-audio-3 library at the commit the card names")
    ap_.add_argument("--snapshot", required=True, help="the Hugging Face cache snapshot directory of the pinned model revision")
    ap_.add_argument("--evidence", help="a directory of evidence JSON files; they are copied to evidence/")
    ap_.add_argument("--numbers", help="a file with the card's numbers table (Markdown); default: a table that says nothing was measured")
    ap_.add_argument("--listening-note")
    args = ap_.parse_args(argv)
    numbers = Path(args.numbers).read_text(encoding="utf-8") if args.numbers else "| Number | Value | Label | Evidence |\n| --- | --- | --- | --- |\n| every number | TODO | TODO | none |"
    try:
        files = assemble(Path(args.out), library=Path(args.library), snapshot=Path(args.snapshot), numbers=numbers,
                         evidence=Path(args.evidence) if args.evidence else None, listening_note=args.listening_note)
    except (BaseError, ap.AudioPackageError) as exc:
        print(f"can_stable_audio_base: {exc}", file=sys.stderr)
        return 2
    print(f"built {len(files)} files in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
