"""scripts/version.py and `python3 -m orchard.version`: show, check, new, release and migrate-pr.

Each test builds a small throwaway repository (its own orchard/__init__.py, changes/ and git history) and passes it to
the command as `root`, so nothing here touches the real repository.
"""
import io
import os
import subprocess
import sys
from pathlib import Path

import pytest

from orchard import version as V
from orchard import version_cli as C

ROOT = Path(__file__).resolve().parent.parent
GIT = ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false"]


def sh(cwd, *args):
    return subprocess.run(list(args), cwd=cwd, capture_output=True, text=True, check=True).stdout


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    (r / "orchard").mkdir(parents=True)
    (r / "orchard" / "__init__.py").write_text('"""pkg"""\n\nRELEASE = "1.0.0"\n')
    (r / "changes").mkdir()
    (r / "changes" / "README.md").write_text("notes\n")
    (r / "docs").mkdir()
    sh(r, "git", "init", "-q", "-b", "main")
    sh(r, "git", "add", "-A")
    sh(r, *GIT, "commit", "-q", "-m", "start")
    return r


def run(repo, *argv):
    out, err = io.StringIO(), io.StringIO()
    try:
        code = C.main(list(argv), root=repo, out=out, err=err)
    except SystemExit as e:                      # argparse exits 2 on a bad option value
        code = e.code
    return code, out.getvalue(), err.getvalue()


def add_frag(repo, slug, target, summary="Does a thing.", extra=""):
    p = repo / "changes" / f"{slug}.toml"
    p.write_text(f'target = "{target}"\nsummary = "{summary}"\n{extra}')
    return p


def tree(repo):
    """Every file under repo (not .git) with its bytes, to compare before and after."""
    return {str(p.relative_to(repo)): p.read_bytes() for p in sorted(repo.rglob("*")) if p.is_file() and ".git" not in p.parts}


# ---- show ----------------------------------------------------------------------------------------

def test_show_lists_release_fragments_and_the_current_version(repo):
    add_frag(repo, "alpha", "1.0.3", "First thing.", "pr = 5\nkind = \"fix\"\n")
    add_frag(repo, "beta", "1.0.2", "Second thing.")
    code, out, _ = run(repo, "show")
    assert code == 0
    assert "RELEASE 1.0.0" in out and "current 1.0.3" in out
    assert "alpha" in out and "1.0.3" in out and "First thing." in out and "#5" in out
    assert "beta" in out and "Second thing." in out


def test_show_marks_a_skipped_fragment_with_its_reason(repo):
    (repo / "changes" / "bad.toml").write_text("not toml = = =")
    code, out, _ = run(repo, "show")
    assert code == 0 and "SKIPPED" in out and "bad" in out and "current 1.0.0" in out


# ---- check ---------------------------------------------------------------------------------------

def test_check_passes_on_a_clean_repository(repo):
    add_frag(repo, "alpha", "1.0.1")
    code, out, _ = run(repo, "check")
    assert code == 0 and "ok" in out.lower()


def test_check_passes_with_no_fragments(repo):
    assert run(repo, "check")[0] == 0


def test_check_fails_on_a_malformed_fragment(repo):
    (repo / "changes" / "bad.toml").write_text('target = "1.2"\nsummary = "x"\n')
    code, out, err = run(repo, "check")
    assert code == 1 and "bad" in (out + err)


def test_check_fails_on_a_target_not_above_the_release_and_says_to_retarget(repo):
    add_frag(repo, "late", "1.0.0")
    code, out, err = run(repo, "check")
    assert code == 1 and "retarget" in (out + err)


def test_check_fails_on_two_fragments_for_one_pr(repo):
    add_frag(repo, "a", "1.0.1", extra="pr = 3\n")
    add_frag(repo, "b", "1.0.2", extra="pr = 3\n")
    assert run(repo, "check")[0] == 1


# ---- new -----------------------------------------------------------------------------------------

def test_new_writes_a_fragment_with_the_next_patch_above_the_current_version(repo):
    add_frag(repo, "alpha", "1.0.7")
    code, out, _ = run(repo, "new", "my-feature", "--summary", "Adds my feature.", "--pr", "12", "--kind", "docs")
    assert code == 0
    (f,) = [x for x in V.load_fragments(repo / "changes") if x.slug == "my-feature"]
    assert (f.target, f.summary, f.pr, f.kind, f.problem) == ("1.0.8", "Adds my feature.", 12, "docs", None)
    assert "changes/my-feature.toml" in out


def test_new_next_patch_compares_numbers_so_0_9_9_goes_to_0_9_10(repo):
    (repo / "orchard" / "__init__.py").write_text('RELEASE = "0.9.9"\n')
    run(repo, "new", "x", "--summary", "Y.")
    assert V.load_fragments(repo / "changes")[0].target == "0.9.10"


def test_new_accepts_an_explicit_target(repo):
    run(repo, "new", "x", "--target", "2.0.0", "--summary", "Y.")
    assert V.load_fragments(repo / "changes")[0].target == "2.0.0"


def test_new_refuses_to_overwrite(repo):
    assert run(repo, "new", "x", "--summary", "First.")[0] == 0
    before = (repo / "changes" / "x.toml").read_bytes()
    code, _, err = run(repo, "new", "x", "--summary", "Second.")
    assert code != 0 and "exists" in err and (repo / "changes" / "x.toml").read_bytes() == before


@pytest.mark.parametrize("slug", ["Bad", "has_underscore", "a" * 61, "", "-", "../x", "x/y", "x.toml"])
def test_new_refuses_a_bad_slug(repo, slug):
    code, _, err = run(repo, "new", slug, "--summary", "Y.")
    assert code != 0
    assert [p.name for p in (repo / "changes").iterdir()] == ["README.md"]


@pytest.mark.parametrize("target", ["1.2", "01.2.3", "1.2.3-rc1", "x"])
def test_new_refuses_a_bad_target(repo, target):
    assert run(repo, "new", "x", "--target", target, "--summary", "Y.")[0] != 0


def test_new_refuses_a_bad_summary_or_kind_or_pr(repo):
    assert run(repo, "new", "x", "--summary", "s" * 301)[0] != 0
    assert run(repo, "new", "x", "--summary", "")[0] != 0
    assert run(repo, "new", "x", "--summary", "Y.", "--kind", "shiny")[0] != 0
    assert run(repo, "new", "x", "--summary", "Y.", "--pr", "0")[0] != 0
    assert [p.name for p in (repo / "changes").iterdir()] == ["README.md"]


def test_new_handles_quotes_and_backslashes_in_a_summary(repo):
    s = 'He said "hi" and used a \\ and a é.'
    assert run(repo, "new", "x", "--summary", s)[0] == 0
    assert V.load_fragments(repo / "changes")[0].summary == s


def test_new_creates_the_changes_directory_when_it_is_missing(repo):
    sh(repo, "git", "rm", "-rq", "changes")
    assert run(repo, "new", "x", "--summary", "Y.")[0] == 0


def test_new_leaves_no_temporary_file_behind(repo):
    run(repo, "new", "x", "--summary", "Y.")
    run(repo, "new", "x", "--summary", "Y.")
    assert sorted(p.name for p in (repo / "changes").iterdir()) == ["README.md", "x.toml"]


# ---- release -------------------------------------------------------------------------------------

def setup_release(repo):
    add_frag(repo, "alpha", "1.0.2", "Adds alpha.", "pr = 5\nkind = \"feature\"\n")
    add_frag(repo, "beta", "1.1.0", "Fixes beta.", "pr = 6\nkind = \"fix\"\n")
    add_frag(repo, "gamma", "1.0.9", "Notes on gamma.", "kind = \"docs\"\n")
    sh(repo, "git", "add", "-A")
    sh(repo, *GIT, "commit", "-q", "-m", "fragments")


def test_release_dry_run_changes_no_file_and_prints_the_plan(repo):
    setup_release(repo)
    before = tree(repo)
    code, out, _ = run(repo, "release", "--dry-run")
    assert code == 0 and tree(repo) == before
    assert "1.1.0" in out and "docs/releases/1.1.0.md" in out and "git tag" in out and "gh release create" in out
    assert "Adds alpha." in out


def test_release_dry_run_is_allowed_with_a_dirty_tree(repo):
    setup_release(repo)
    (repo / "scratch.txt").write_text("x")
    assert run(repo, "release", "--dry-run")[0] == 0


def test_release_refuses_a_dirty_tree(repo):
    setup_release(repo)
    (repo / "scratch.txt").write_text("x")
    before = tree(repo)
    code, _, err = run(repo, "release")
    assert code != 0 and "dirty" in err and tree(repo) == before


def test_release_refuses_a_tree_with_a_staged_change(repo):
    setup_release(repo)
    (repo / "docs" / "n.md").write_text("x")
    sh(repo, "git", "add", "docs/n.md")
    assert run(repo, "release")[0] != 0


def test_release_writes_the_notes_sets_release_and_consumes_the_fragments(repo):
    setup_release(repo)
    code, out, _ = run(repo, "release")
    assert code == 0, out
    assert V.read_release(repo / "orchard" / "__init__.py") == "1.1.0"
    assert V.current(repo / "changes", release="1.1.0") == "1.1.0"
    assert V.load_fragments(repo / "changes") == [] and (repo / "changes" / "README.md").exists()
    notes = (repo / "docs" / "releases" / "1.1.0.md").read_text()
    assert notes.startswith("# 1.1.0") and "Adds alpha." in notes and "Fixes beta." in notes and "Notes on gamma." in notes
    assert "#5" in notes and "#6" in notes
    assert notes.index("Adds alpha.") < notes.index("Fixes beta.") < notes.index("Notes on gamma.")     # feature, fix, docs
    assert "git tag" in out and "v1.1.0" in out and "gh release create" in out
    assert sh(repo, "git", "tag", "--list") == "" and sh(repo, "git", "log", "--oneline").count("\n") == 2   # no tag, no commit


def test_release_notes_carry_the_date(repo):
    setup_release(repo)
    run(repo, "release")
    import re
    assert re.search(r"\d{4}-\d{2}-\d{2}", (repo / "docs" / "releases" / "1.1.0.md").read_text())


def test_release_keeps_the_rest_of_init_py_unchanged(repo):
    setup_release(repo)
    before = (repo / "orchard" / "__init__.py").read_text()
    run(repo, "release")
    assert (repo / "orchard" / "__init__.py").read_text() == before.replace('"1.0.0"', '"1.1.0"')


def test_release_refuses_when_there_is_nothing_to_release(repo):
    code, _, err = run(repo, "release", "--dry-run")
    assert code != 0 and "nothing" in err


def test_release_refuses_when_check_finds_a_problem_and_changes_nothing(repo):
    setup_release(repo)
    (repo / "changes" / "bad.toml").write_text("= =")
    sh(repo, "git", "add", "-A")
    sh(repo, *GIT, "commit", "-q", "-m", "bad")
    before = tree(repo)
    code, _, err = run(repo, "release")
    assert code != 0 and tree(repo) == before


def test_release_refuses_to_overwrite_existing_notes(repo):
    setup_release(repo)
    (repo / "docs" / "releases").mkdir()
    (repo / "docs" / "releases" / "1.1.0.md").write_text("old")
    sh(repo, "git", "add", "-A")
    sh(repo, *GIT, "commit", "-q", "-m", "notes")
    code, _, err = run(repo, "release")
    assert code != 0 and "refusing to overwrite" in err and (repo / "docs" / "releases" / "1.1.0.md").read_text() == "old"
    assert V.load_fragments(repo / "changes")


def test_release_in_a_directory_that_is_not_a_git_repository_refuses(tmp_path):
    (tmp_path / "orchard").mkdir()
    (tmp_path / "orchard" / "__init__.py").write_text('RELEASE = "1.0.0"\n')
    (tmp_path / "changes").mkdir()
    add_frag(tmp_path, "a", "1.0.1")
    assert run(tmp_path, "release")[0] != 0


# ---- migrate-pr ----------------------------------------------------------------------------------

def old_style_branch(repo, name="old-feature", version="1.0.4", subject="Adds the old feature (version 1.0.4)"):
    sh(repo, "git", "checkout", "-q", "-b", name)
    (repo / "orchard" / "__init__.py").write_text(f'"""pkg"""\n\n__version__ = "{version}"\n')
    sh(repo, "git", "add", "-A")
    sh(repo, *GIT, "commit", "-q", "-m", subject)
    sh(repo, "git", "checkout", "-q", "main")


def test_migrate_pr_prints_the_fragment_and_writes_nothing(repo):
    old_style_branch(repo)
    before = tree(repo)
    code, out, _ = run(repo, "migrate-pr", "--from-ref", "old-feature")
    assert code == 0 and tree(repo) == before
    assert "changes/old-feature.toml" in out and 'target = "1.0.4"' in out and "Adds the old feature" in out
    (summary_line,) = [l for l in out.splitlines() if l.startswith("summary = ")]
    assert summary_line == 'summary = "Adds the old feature"'


def test_migrate_pr_strips_a_remote_prefix_and_takes_a_pr_number(repo):
    old_style_branch(repo, "feat/Odd_Name")
    code, out, _ = run(repo, "migrate-pr", "--from-ref", "feat/Odd_Name", "--pr", "77")
    assert code == 0 and "changes/feat-odd-name.toml" in out and "pr = 77" in out


def test_migrate_pr_reads_a_remote_tracking_ref_and_names_the_slug_without_origin(repo):
    old_style_branch(repo, "feat-x")
    sh(repo, "git", "update-ref", "refs/remotes/origin/feat-x", "feat-x")
    code, out, _ = run(repo, "migrate-pr", "--from-ref", "origin/feat-x")
    assert code == 0 and "changes/feat-x.toml" in out and "origin-feat-x" not in out


def test_migrate_pr_reports_a_branch_that_already_uses_fragments(repo):
    sh(repo, "git", "checkout", "-q", "-b", "new-style")
    add_frag(repo, "new-style", "1.0.5")
    sh(repo, "git", "add", "-A")
    sh(repo, *GIT, "commit", "-q", "-m", "x")
    sh(repo, "git", "checkout", "-q", "main")
    code, out, err = run(repo, "migrate-pr", "--from-ref", "new-style")
    assert code == 0 and "already" in (out + err)


def test_migrate_pr_with_an_unknown_ref_fails(repo):
    code, _, err = run(repo, "migrate-pr", "--from-ref", "nope")
    assert code != 0 and err


def test_migrate_pr_with_a_malformed_version_at_the_ref_fails(repo):
    old_style_branch(repo, "weird", "1.2")
    assert run(repo, "migrate-pr", "--from-ref", "weird")[0] != 0


# ---- the commands as run from a shell ------------------------------------------------------------

def test_the_script_and_the_module_give_the_same_show_output():
    a = subprocess.run([sys.executable, str(ROOT / "scripts" / "version.py"), "show"], capture_output=True, text=True, cwd="/", timeout=60)
    b = subprocess.run([sys.executable, "-m", "orchard.version", "show"], capture_output=True, text=True, cwd=ROOT, timeout=60)
    assert a.returncode == b.returncode == 0 and a.stdout == b.stdout and a.stdout


def test_the_script_check_passes_on_this_repository():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "version.py"), "check"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr


def test_a_bad_command_line_exits_2():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "version.py"), "frobnicate"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 2


def test_atomic_write_goes_through_a_temp_file_fsync_and_replace(tmp_path, monkeypatch):
    calls = []
    real_fsync, real_replace = os.fsync, os.replace
    monkeypatch.setattr(os, "fsync", lambda fd: (calls.append("fsync"), real_fsync(fd))[1])
    monkeypatch.setattr(os, "replace", lambda a, b: (calls.append(("replace", Path(a).parent == Path(b).parent, Path(a) != Path(b))), real_replace(a, b))[1])
    C.atomic_write(tmp_path / "f.txt", "hello")
    assert (tmp_path / "f.txt").read_text() == "hello"
    assert calls[0] == "fsync" and ("replace", True, True) in calls
    assert [p.name for p in tmp_path.iterdir()] == ["f.txt"]


def test_atomic_write_that_fails_leaves_the_old_file_and_no_temp_file(tmp_path, monkeypatch):
    (tmp_path / "f.txt").write_text("old")
    monkeypatch.setattr(os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError):
        C.atomic_write(tmp_path / "f.txt", "new")
    assert (tmp_path / "f.txt").read_text() == "old" and [p.name for p in tmp_path.iterdir()] == ["f.txt"]
