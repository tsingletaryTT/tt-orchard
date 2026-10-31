"""The Stable Audio 3 can: the base-directory script (scripts/can_stable_audio_base.py), the card template and the spec (config/can.stable-audio-3.toml).

The model library, the model's license and the weights are not in this repository, so the tests build stand-ins: a small git repository for the library,
a snapshot directory with a license text that has the paragraphs the card quotes. Staging with the real `tt-model package-thin` is skipped when
`tt-model` is not installed."""
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from orchard import audio_package as ap
from orchard import can

ROOT = Path(__file__).resolve().parent.parent
spec_ = importlib.util.spec_from_file_location("can_stable_audio_base_under_test", ROOT / "scripts" / "can_stable_audio_base.py")
base_mod = importlib.util.module_from_spec(spec_)
sys.modules["can_stable_audio_base_under_test"] = base_mod
spec_.loader.exec_module(base_mod)
HAVE_TT_MODEL = shutil.which("tt-model") is not None
REV = base_mod.MODEL_REVISION

# the agreement's paragraphs the card quotes, as the real text writes them (curly quotes and all)
LICENSE = """STABILITY AI COMMUNITY LICENSE AGREEMENT

2. RESEARCH & NON-COMMERCIAL USE LICENSE

Subject to the terms of this Agreement, Stability AI grants You a non-exclusive license to use the Stability AI Materials for any Research or Non-Commercial Purpose. Other words.

3. COMMERCIAL USE LICENSE

If You are using or distributing the Stability AI Materials for a Commercial Purpose, You must register with Stability AI at (https://stability.ai/community-license).
If at any time You or Your Affiliate(s), either individually or in aggregate, generate more than USD $1,000,000 in annual revenue, any licenses granted to You under this Agreement shall terminate. You must request a license from Stability AI at (https://stability.ai/enterprise) , which Stability AI may grant to You in its sole discretion.

4. GENERAL TERMS

a.  Distribution & Attribution. If You distribute or make available the Stability AI Materials or a Derivative Work to a third party, You shall: (i) provide a copy of this Agreement to that third party, (ii) retain the following attribution notice within a "Notice" text file distributed as a part of such copies: "This Stability AI Model is licensed under the Stability AI Community License, Copyright ©  Stability AI Ltd. All Rights Reserved”, and (iii) prominently display “Powered by Stability AI” on a related website, user interface, blogpost, about page, or product documentation.  Next sentence.
b.  Use Restrictions. Your use of the Stability AI Materials and Derivative Works, including any output or results of the Stability AI Materials or Derivative Works, must comply with applicable laws. Furthermore, You will not use the Stability AI Materials or Derivative Works to create or improve any foundational generative AI model (excluding the Models or Derivative Works).
"""


def git(cwd, *args):
    return subprocess.run(["git", "-C", str(cwd), "-c", "user.name=t", "-c", "user.email=t@example.com", *args], capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def library(tmp_path):
    lib = tmp_path / "lib"
    (lib / "stable_audio_3" / "sub").mkdir(parents=True)
    (lib / "stable_audio_3" / "__init__.py").write_text("X = 1\n")
    (lib / "stable_audio_3" / "sub" / "m.py").write_text("Y = 2\n")
    (lib / "LICENSE").write_text("MIT License\n")
    git(lib, "init", "-q")
    git(lib, "add", "-A")
    git(lib, "commit", "-q", "-m", "x")
    return lib, git(lib, "rev-parse", "HEAD")


@pytest.fixture
def snapshot(tmp_path):
    snap = tmp_path / "hub" / "snapshots" / REV
    blobs = tmp_path / "hub" / "blobs"
    snap.mkdir(parents=True)
    blobs.mkdir()
    for name, text in (("LICENSE.md", LICENSE), ("NOTICE", "Gemma is provided under the Gemma Terms of Use\n"), ("LICENSE_GEMMA.md", "gemma terms\n")):
        (blobs / name).write_text(text)
        (snap / name).symlink_to(blobs / name)            # a cache snapshot holds links to blobs
    return snap


NUMBERS = "| Number | Value | Label | Evidence |\n| --- | --- | --- | --- |\n| generation time, 10 s clip | 1.0 s | measured | `evidence/session-s1.json` |"


def evidence_dir(tmp_path):
    """A directory with every evidence file the spec names (empty JSON objects)."""
    ev = tmp_path / "evidence"
    if not ev.exists():
        ev.mkdir()
        for name in base_mod.EVIDENCE_FILES:
            (ev / name).write_text("{}\n")
    return ev


def build(tmp_path, library, snapshot, **kw):
    lib, commit = library
    out = tmp_path / "base"
    files = base_mod.assemble(out, library=lib, snapshot=snapshot, numbers=NUMBERS, library_commit=commit, evidence=evidence_dir(tmp_path), **kw)
    return out, files


# ---- the base directory

def test_assemble_builds_every_file_the_spec_names(tmp_path, library, snapshot):
    out, files = build(tmp_path, library, snapshot)
    spec = can.load_spec(ROOT / "config" / "can.stable-audio-3.toml", base=out)
    assert spec.kind == "tt-dit-server" and spec.app == "stable_audio_asgi:app" and spec.check.mode == "audio-job"
    assert {"stable_audio_asgi.py", "stable_audio_jobs.py", "orchard/resident.py", "orchard/__init__.py", "stable_audio_3/__init__.py",
            "stable_audio_3/sub/m.py"} <= {d for _, d in spec.extra_code}
    assert "README.md" in files and (out / "LICENSE.md").read_text() == LICENSE
    assert (out / "runtime/orchard/__init__.py").read_text() == ap.RUNTIME_INIT


def test_the_library_must_be_the_pinned_commit_and_clean(tmp_path, library, snapshot):
    lib, commit = library
    with pytest.raises(base_mod.BaseError, match="names"):
        base_mod.assemble(tmp_path / "b1", library=lib, snapshot=snapshot, numbers=NUMBERS, library_commit="0" * 40, evidence=evidence_dir(tmp_path))
    (lib / "stable_audio_3" / "__init__.py").write_text("X = 2\n")
    with pytest.raises(base_mod.BaseError, match="uncommitted"):
        base_mod.assemble(tmp_path / "b2", library=lib, snapshot=snapshot, numbers=NUMBERS, library_commit=commit, evidence=evidence_dir(tmp_path))


def test_the_snapshot_directory_must_be_named_for_the_pinned_revision(tmp_path, library, snapshot):
    other = snapshot.parent / ("f" * 40)
    snapshot.rename(other)
    with pytest.raises(base_mod.BaseError, match="pinned revision"):
        build(tmp_path, library, other)


def test_a_license_that_lacks_a_quoted_paragraph_is_refused(tmp_path, library, snapshot):
    blob = snapshot.parent.parent / "blobs" / "LICENSE.md"
    blob.write_text(LICENSE.replace("Your use of the Stability AI Materials and Derivative Works", "Your usage of the Materials"))
    with pytest.raises(base_mod.BaseError, match="restrictions|attribution"):
        build(tmp_path, library, snapshot)


def test_a_link_inside_the_library_is_refused_and_leaves_no_base(tmp_path, library, snapshot):
    lib, commit = library
    (lib / "stable_audio_3" / "evil.py").symlink_to(lib / "LICENSE")
    git(lib, "add", "-A")
    git(lib, "commit", "-q", "-m", "link")
    with pytest.raises(base_mod.BaseError, match="link"):
        base_mod.assemble(tmp_path / "b", library=lib, snapshot=snapshot, numbers=NUMBERS, library_commit=git(lib, "rev-parse", "HEAD"), evidence=evidence_dir(tmp_path))
    assert not (tmp_path / "b").exists()


def test_a_missing_evidence_file_is_refused_and_leaves_no_base(tmp_path, library, snapshot):
    lib, commit = library
    empty = tmp_path / "empty-evidence"
    empty.mkdir()
    with pytest.raises(base_mod.BaseError, match="evidence directory has no"):
        base_mod.assemble(tmp_path / "b", library=lib, snapshot=snapshot, numbers=NUMBERS, library_commit=commit, evidence=empty)
    assert not (tmp_path / "b").exists()


def test_an_existing_output_directory_is_refused(tmp_path, library, snapshot):
    build(tmp_path, library, snapshot)
    with pytest.raises(base_mod.BaseError, match="already exists"):
        build(tmp_path, library, snapshot)


def test_the_notice_holds_the_attribution_sentence_the_agreement_asks_for(tmp_path, library, snapshot):
    out, _ = build(tmp_path, library, snapshot)
    notice = (out / "NOTICE").read_text()
    assert 'This Stability AI Model is licensed under the Stability AI Community License, Copyright ©  Stability AI Ltd. All Rights Reserved' in notice
    assert "Gemma Terms of Use" in notice


# ---- the card

def test_the_card_quotes_the_agreement_word_for_word_and_displays_powered_by(tmp_path, library, snapshot):
    out, _ = build(tmp_path, library, snapshot)
    card = (out / "README.md").read_text()
    flat = ap._flat(LICENSE)
    for name, start, end in (*ap.QUOTES, *base_mod.EXTRA_QUOTES):
        q = ap.quote(LICENSE, start, end)
        assert f"> {q}" in card and q in flat, name
    assert "Powered by Stability AI" in card


def test_the_rendered_card_passes_its_own_checks(tmp_path, library, snapshot):
    out, _ = build(tmp_path, library, snapshot)
    assert base_mod.card_problems((out / "README.md").read_text(), LICENSE.encode()) == []


def good_card(tmp_path, library, snapshot):
    out, _ = build(tmp_path, library, snapshot)
    return (out / "README.md").read_text()


@pytest.mark.parametrize("edit,words", [
    (lambda c: c.replace("- tt-model-cache\n", ""), "tt-model-cache"),
    (lambda c: c.replace("- thin\n", "- thin\n- tt-model-catalog\n"), "tt-model-catalog"),
    (lambda c: c.replace("license: other", "license: mit"), "license: other"),
    (lambda c: c.replace("pipeline_tag: text-to-audio", "pipeline_tag: text-generation"), "pipeline_tag"),
    (lambda c: c.replace("## Expected performance", "## Numbers"), "Expected performance"),
    (lambda c: c.replace("## Limitations", "## Known issues"), "Limitations"),
    (lambda c: c.replace("Powered by Stability AI", "Made with Stability"), "Powered by Stability AI"),
    (lambda c: c + "\n{{NUMBERS}}\n", "unresolved placeholder"),
    (lambda c: c + "\nSee `evidence/made-up.json`.\n", "evidence/made-up.json"),
    (lambda c: c.replace("> Subject to the terms of this Agreement, Stability AI grants You", "> Stability AI grants everyone"), "not in the license copy"),
])
def test_the_card_check_reports_each_fault(tmp_path, library, snapshot, edit, words):
    problems = base_mod.card_problems(edit(good_card(tmp_path, library, snapshot)), LICENSE.encode())
    assert any(words in p for p in problems), problems


def test_the_template_has_no_unfinished_marker_and_no_machine_path():
    text = (ROOT / "pkgsrc" / "can" / "stable-audio-3-README.md").read_text()
    for bad in ("TODO", "<<TBD", "/home/", "/mnt/"):
        assert bad not in text, bad


def test_the_card_never_calls_the_output_good_or_the_package_fast():
    text = (ROOT / "pkgsrc" / "can" / "stable-audio-3-README.md").read_text().lower()
    for claim in ("high-quality", "high quality", "state-of-the-art", "production-ready", "verified", "blazing", "sounds great", "sounds good"):
        assert claim not in text, claim


# ---- the spec

def test_the_spec_pins_every_package_exactly_and_keeps_to_its_port_range():
    import tomllib
    raw = tomllib.loads((ROOT / "config" / "can.stable-audio-3.toml").read_text())
    assert can.requirement_problems("\n".join(raw["can"]["requirements"])) == []
    assert 20170 <= raw["can"]["port"] <= 20179
    assert raw["weights"]["revision"] == REV and raw["weights"]["repo"] == "stabilityai/stable-audio-3-small-music"


@pytest.mark.skipif(not HAVE_TT_MODEL, reason="tt-model is not installed")
def test_the_whole_base_stages_through_the_real_package_thin_and_passes_the_gate(tmp_path, library, snapshot):
    base, _ = build(tmp_path, library, snapshot)
    spec = can.load_spec(ROOT / "config" / "can.stable-audio-3.toml", base=base)
    rec = can.stage(spec, tmp_path / "out" / spec.name)
    pkg = tmp_path / "out" / spec.name
    assert rec["kind"] == "tt-dit-server" and can.gate(pkg, spec) == []
    names = {p.relative_to(pkg).as_posix() for p in pkg.rglob("*") if p.is_file()}
    assert {"README.md", "LICENSE.md", "NOTICE", "extra_code/stable_audio_asgi.py", "extra_code/stable_audio_3/sub/m.py",
            "extra_code/orchard/skills/audio-chip-templates/audio_runtimes.py", "tools/audio_client.py", "SHA256SUMS"} <= names
    assert not any(n.startswith(("venv/", ".hf/", ".tt_cache/")) or n.endswith((".safetensors", ".bin")) for n in names)
