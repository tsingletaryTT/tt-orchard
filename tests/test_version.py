"""The version is computed: RELEASE (a literal in orchard/__init__.py) or the highest target in changes/.

Every guard in orchard/version.py has a test here. Fragments are read at runtime, so a bad one must be skipped
with a reason and never raise; the repository check at the bottom makes the TESTS fail on any bad fragment.
"""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import orchard
from orchard import version as V

ROOT = Path(__file__).resolve().parent.parent


def frag(d: Path, slug: str, target="9.9.9", summary="Does a thing.", extra="", raw=None) -> Path:
    """Write changes/<slug>.toml; `raw` replaces the whole text."""
    d.mkdir(exist_ok=True)
    p = d / f"{slug}.toml"
    if raw is not None:
        p.write_bytes(raw if isinstance(raw, bytes) else raw.encode())
    else:
        p.write_text(f'target = "{target}"\nsummary = "{summary}"\n{extra}')
    return p


@pytest.fixture
def ch(tmp_path):
    return tmp_path / "changes"


# ---- syntax and numeric comparison ---------------------------------------------------------------

@pytest.mark.parametrize("s,t", [("0.0.0", (0, 0, 0)), ("1.2.3", (1, 2, 3)), ("0.9.36", (0, 9, 36)),
                                 ("999999.999999.999999", (999999, 999999, 999999))])
def test_a_valid_version_parses_to_a_tuple_of_numbers(s, t):
    assert V.parse(s) == t and V.fmt(t) == s


@pytest.mark.parametrize("s", ["01.2.3", "1.02.3", "1.2.03", "1.2", "1.2.3.4", "1.2.3-rc1", "1.2.3\n", " 1.2.3", "1.2.3 ",
                               "", "v1.2.3", "1.2.x", "-1.2.3", "1000000.0.0", "0.1000000.0", "0.0.1000000",
                               "١.٢.٣", "1.2.3\x00", "+1.2.3"])
def test_a_malformed_version_is_refused(s):
    assert V.parse(s) is None


@pytest.mark.parametrize("not_a_string", [None, 1, 1.2, True, ["1.2.3"], b"1.2.3"])
def test_a_version_that_is_not_a_string_is_refused(not_a_string):
    assert V.parse(not_a_string) is None


def test_versions_compare_as_numbers_not_text(ch):
    assert V.parse("0.9.10") > V.parse("0.9.9")
    frag(ch, "ten", "0.9.10")
    frag(ch, "nine", "0.9.9")
    assert V.current(ch, release="0.9.8") == "0.9.10"


# ---- the computed current version ----------------------------------------------------------------

def test_with_no_changes_directory_the_current_version_is_the_release(tmp_path):
    assert V.current(tmp_path / "absent", release="1.0.0") == "1.0.0"


def test_with_an_empty_changes_directory_the_current_version_is_the_release(ch):
    ch.mkdir()
    assert V.current(ch, release="1.0.0") == "1.0.0"


def test_the_current_version_is_the_highest_target(ch):
    frag(ch, "a", "1.0.2")
    frag(ch, "b", "1.0.5")
    frag(ch, "c", "1.0.3")
    assert V.current(ch, release="1.0.0") == "1.0.5"


def test_two_fragments_may_claim_the_same_target(ch):
    frag(ch, "a", "1.0.2")
    frag(ch, "b", "1.0.2")
    assert V.current(ch, release="1.0.0") == "1.0.2"
    assert V.problems(ch, release="1.0.0") == []


def test_a_fragment_below_the_release_does_not_lower_or_raise_the_version(ch):
    frag(ch, "old", "0.9.1")
    assert V.current(ch, release="1.0.0") == "1.0.0"


def test_check_flags_a_fragment_not_above_the_release_and_says_to_retarget_it(ch):
    frag(ch, "old", "1.0.0")
    (msg,) = V.problems(ch, release="1.0.0")
    assert "old" in msg and "retarget" in msg


# ---- malformed fragments: skipped at runtime with a reason, problems for check --------------------

def bad_cases(ch: Path):
    """name -> function that creates one bad fragment named `bad` (and returns nothing)."""
    def link():
        real = ch.parent / "real.toml"
        real.write_text('target = "9.0.0"\nsummary = "x"\n')
        ch.mkdir(exist_ok=True)
        os.symlink(real, ch / "bad.toml")

    def directory():
        (ch / "bad.toml").mkdir(parents=True)

    def fifo():
        ch.mkdir(exist_ok=True)
        os.mkfifo(ch / "bad.toml")

    return {
        "directory": directory,
        "symlink": link,
        "fifo": fifo,
        "five_kib": lambda: frag(ch, "bad", summary="x", extra="# " + "y" * 5000 + "\n"),
        "invalid_utf8": lambda: frag(ch, "bad", raw=b'target = "9.0.0"\nsummary = "\xff\xfe"\n'),
        "toml_list": lambda: frag(ch, "bad", raw="[1, 2, 3]\n"),
        "toml_syntax": lambda: frag(ch, "bad", raw='target = "9.0.0\n'),
        "bool_target": lambda: frag(ch, "bad", raw='target = true\nsummary = "x"\n'),
        "int_target": lambda: frag(ch, "bad", raw='target = 9\nsummary = "x"\n'),
        "leading_zero": lambda: frag(ch, "bad", "01.2.3"),
        "two_parts": lambda: frag(ch, "bad", "1.2"),
        "prerelease": lambda: frag(ch, "bad", "1.2.3-rc1"),
        "newline": lambda: frag(ch, "bad", raw='target = "1.2.3\\n"\nsummary = "x"\n'),
        "huge_number": lambda: frag(ch, "bad", "1000000.0.0"),
        "no_target": lambda: frag(ch, "bad", raw='summary = "x"\n'),
        "no_summary": lambda: frag(ch, "bad", raw='target = "9.0.0"\n'),
        "empty_summary": lambda: frag(ch, "bad", summary=""),
        "long_summary": lambda: frag(ch, "bad", summary="s" * 301),
        "bool_summary": lambda: frag(ch, "bad", raw='target = "9.0.0"\nsummary = false\n'),
        "unknown_key": lambda: frag(ch, "bad", extra='colour = "red"\n'),
        "bool_pr": lambda: frag(ch, "bad", extra="pr = true\n"),
        "string_pr": lambda: frag(ch, "bad", extra='pr = "12"\n'),
        "zero_pr": lambda: frag(ch, "bad", extra="pr = 0\n"),
        "unknown_kind": lambda: frag(ch, "bad", extra='kind = "shiny"\n'),
        "bad_slug_upper": lambda: frag(ch, "Bad"),
        "bad_slug_underscore": lambda: frag(ch, "bad_slug"),
        "slug_too_long": lambda: frag(ch, "a" * 61),
        "not_toml_file": lambda: (ch.mkdir(exist_ok=True), (ch / "notes.txt").write_text("hi")),
    }


CASES = list(bad_cases(Path("x")))


@pytest.mark.parametrize("case", CASES)
def test_a_bad_fragment_is_skipped_and_recorded_and_never_raises(case, ch):
    bad_cases(ch)[case]()
    frag(ch, "good", "9.0.1")
    assert V.current(ch, release="9.0.0") == "9.0.1"          # the good one still counts, the bad one is ignored
    assert V.SKIPPED, f"{case}: nothing was recorded as skipped"
    assert all(isinstance(name, str) and isinstance(why, str) and why for name, why in V.SKIPPED)
    assert V.problems(ch, release="9.0.0"), f"{case}: check reports nothing"


@pytest.mark.parametrize("case", CASES)
def test_a_bad_fragment_never_raises_even_when_it_is_the_only_file(case, ch):
    bad_cases(ch)[case]()
    assert V.current(ch, release="9.0.0") == "9.0.0"


def test_a_target_above_a_skipped_fragment_is_not_used(ch):
    frag(ch, "bad", extra='colour = "red"\n', target="50.0.0")
    assert V.current(ch, release="1.0.0") == "1.0.0"


def test_a_fifo_does_not_block_a_reader(tmp_path):
    ch = tmp_path / "changes"
    ch.mkdir()
    os.mkfifo(ch / "pipe.toml")
    code = ("import sys; from pathlib import Path; from orchard import version as V; "
            "print(V.current(Path(sys.argv[1]), release='1.0.0'), len(V.SKIPPED))")
    r = subprocess.run([sys.executable, "-c", code, str(ch)], capture_output=True, text=True, timeout=20, cwd=ROOT)
    assert r.returncode == 0 and r.stdout.split() == ["1.0.0", "1"], r.stderr


def test_a_file_of_exactly_4_kib_is_read_and_one_byte_more_is_not(ch):
    head = 'target = "9.0.1"\nsummary = "ok"\n#'
    ch.mkdir()
    (ch / "exact.toml").write_text(head + "y" * (4096 - len(head)))
    assert (ch / "exact.toml").stat().st_size == 4096
    assert V.current(ch, release="9.0.0") == "9.0.1"
    (ch / "exact.toml").write_text(head + "y" * (4097 - len(head)))
    assert V.current(ch, release="9.0.0") == "9.0.0"


def test_a_summary_of_exactly_300_characters_is_accepted(ch):
    frag(ch, "ok", "9.0.1", summary="s" * 300)
    assert V.current(ch, release="9.0.0") == "9.0.1"


def test_a_slug_of_exactly_60_characters_is_accepted(ch):
    frag(ch, "a" * 60, "9.0.1")
    assert V.current(ch, release="9.0.0") == "9.0.1"


def test_a_directory_with_more_than_500_files_reads_500_and_reports_the_rest(ch):
    ch.mkdir()
    for i in range(505):
        (ch / f"f{i:04d}.toml").write_text('target = "9.0.1"\nsummary = "x"\n')
    frags = V.load_fragments(ch)
    assert sum(1 for f in frags if f.problem is None) == 500
    assert any("500" in p for p in V.problems(ch, release="9.0.0"))
    assert V.current(ch, release="9.0.0") == "9.0.1"


def test_readme_in_changes_is_not_a_fragment(ch):
    ch.mkdir()
    (ch / "README.md").write_text("notes")
    V.current(ch, release="1.0.0")
    assert V.problems(ch, release="1.0.0") == [] and V.SKIPPED == []


def test_two_fragments_for_one_pr_number_are_a_problem(ch):
    frag(ch, "a", "1.0.2", extra="pr = 7\n")
    frag(ch, "b", "1.0.3", extra="pr = 7\n")
    (msg,) = V.problems(ch, release="1.0.0")
    assert "7" in msg and "a" in msg and "b" in msg


def test_optional_fields_are_read(ch):
    frag(ch, "a", "1.0.2", extra='pr = 7\nkind = "fix"\n')
    (f,) = V.load_fragments(ch)
    assert (f.slug, f.target, f.summary, f.pr, f.kind, f.problem) == ("a", "1.0.2", "Does a thing.", 7, "fix", None)


def test_a_placeholder_summary_is_a_problem_for_check_but_still_counts_at_runtime(ch):
    frag(ch, "a", "1.0.2", summary="TODO: describe this change in one sentence.")
    assert V.current(ch, release="1.0.0") == "1.0.2"
    assert any("TODO" in p for p in V.problems(ch, release="1.0.0"))


# ---- the release literal and orchard.__version__ --------------------------------------------------

def test_the_release_literal_is_read_without_importing_the_package(tmp_path):
    p = tmp_path / "__init__.py"
    p.write_text('"""doc"""\n\nRELEASE = "3.4.5"\n')
    assert V.read_release(p) == "3.4.5"


def test_the_release_in_the_repository_matches_what_the_package_reports():
    assert V.parse(orchard.RELEASE) is not None
    assert V.read_release(ROOT / "orchard" / "__init__.py") == orchard.RELEASE == V.RELEASE


def test_dunder_version_equals_the_computed_current_version():
    assert orchard.__version__ == V.current()
    assert V.parse(orchard.__version__) >= V.parse(orchard.RELEASE)


def test_an_installed_copy_without_a_changes_directory_reports_the_release(tmp_path):
    """Copy the package somewhere with no changes/ beside it and import it there."""
    import shutil
    shutil.copytree(ROOT / "orchard", tmp_path / "orchard", ignore=shutil.ignore_patterns("__pycache__", "skills", "package_templates"))
    code = "import orchard; print(orchard.__version__, orchard.RELEASE)"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30, cwd=tmp_path,
                       env={"PYTHONPATH": str(tmp_path), "PATH": os.environ.get("PATH", "")})
    assert r.returncode == 0, r.stderr
    shown, release = r.stdout.split()
    assert shown == release


def test_python_dash_m_orchard_version_runs_without_a_warning():
    r = subprocess.run([sys.executable, "-m", "orchard.version", "show"], capture_output=True, text=True, timeout=30, cwd=ROOT)
    assert r.returncode == 0 and "RELEASE" in r.stdout and "Warning" not in r.stderr, r.stderr


# ---- the repository itself -----------------------------------------------------------------------

def test_every_fragment_in_the_repository_passes_check():
    assert V.problems(V.CHANGES_DIR, release=V.RELEASE) == []


def test_no_fragment_in_the_repository_is_at_or_below_the_release():
    for f in V.load_fragments(V.CHANGES_DIR):
        assert f.problem is None, (f.slug, f.problem)
        assert V.parse(f.target) > V.parse(V.RELEASE), (
            f"changes/{f.slug}.toml targets {f.target}, which is not above RELEASE {V.RELEASE}: retarget it")


# ---- pip metadata --------------------------------------------------------------------------------

def _pip_works(py: str) -> bool:
    r = subprocess.run([py, "-m", "pip", "--version"], capture_output=True, text=True)
    return r.returncode == 0


@pytest.mark.parametrize("editable", [True, False], ids=["editable", "plain"])
def test_pip_metadata_reports_the_release_literal(tmp_path, editable):
    if not _pip_works(sys.executable):
        pytest.skip("no usable pip for this interpreter")
    venv = tmp_path / "venv"
    r = subprocess.run([sys.executable, "-m", "venv", str(venv)], capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"cannot create a virtualenv: {r.stderr.strip()[-200:]}")
    py = str(venv / "bin" / "python")
    src = tmp_path / "src"                                       # build from a copy so no build directory lands in the repo
    import shutil
    shutil.copytree(ROOT, src, ignore=shutil.ignore_patterns(".git", "tests", "site", "docs", "__pycache__", "*.egg-info", "build", ".claude"))
    cmd = [py, "-m", "pip", "install", "--no-deps", "--disable-pip-version-check", "--no-input"] + (["-e"] if editable else []) + [str(src)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        tail = (r.stdout + r.stderr)[-600:]
        if "No matching distribution" in tail or "Could not find a version" in tail or "Connection" in tail or "Temporary failure" in tail:
            pytest.skip(f"pip could not fetch the build backend (no network): {tail[-200:]}")
        raise AssertionError(tail)
    shown = subprocess.run([py, "-m", "pip", "show", "tt-orchard"], capture_output=True, text=True).stdout
    assert f"Version: {V.read_release(ROOT / 'orchard' / '__init__.py')}" in shown, shown
