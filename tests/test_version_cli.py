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


# ---- tag-check and the tagging sequence that `release` prints -------------------------------------

def released_repo(repo):
    """A repository on main where `release` has run and its result is committed (the state after the release PR merged)."""
    setup_release(repo)
    assert run(repo, "release")[0] == 0
    sh(repo, "git", "add", "-A")
    sh(repo, *GIT, "commit", "-q", "-m", "Release 1.1.0")
    return sh(repo, "git", "rev-parse", "HEAD").strip()


def test_tag_check_passes_on_the_merged_release_commit_and_prints_the_exact_tag_command(repo):
    sha = released_repo(repo)
    code, out, err = run(repo, "tag-check", "1.1.0")
    assert code == 0, err
    assert sha in out and f"git tag -a v1.1.0 -m 'Release 1.1.0' {sha}" in out and "git push origin v1.1.0" in out
    assert sh(repo, "git", "tag", "--list") == ""                  # it only checks


def test_tag_check_accepts_a_leading_v(repo):
    released_repo(repo)
    assert run(repo, "tag-check", "v1.1.0")[0] == 0


@pytest.mark.parametrize("bad", ["1.1", "01.1.0", "x", "1.1.0-rc1"])
def test_tag_check_refuses_a_malformed_version(repo, bad):
    released_repo(repo)
    code, _, err = run(repo, "tag-check", bad)
    assert code == 2 and err


def test_tag_check_refuses_a_version_that_is_not_release(repo):
    released_repo(repo)
    code, _, err = run(repo, "tag-check", "1.0.5")
    assert code == 1 and "RELEASE" in err


def test_tag_check_refuses_while_a_fragment_remains(repo):
    released_repo(repo)
    add_frag(repo, "late", "1.1.1")
    sh(repo, "git", "add", "-A")
    sh(repo, *GIT, "commit", "-q", "-m", "late")
    code, _, err = run(repo, "tag-check", "1.1.0")
    assert code == 1 and "fragment" in err and "late" in err


def test_tag_check_refuses_a_dirty_tree(repo):
    released_repo(repo)
    (repo / "scratch.txt").write_text("x")
    code, _, err = run(repo, "tag-check", "1.1.0")
    assert code == 1 and "dirty" in err


def test_tag_check_refuses_when_the_tag_exists(repo):
    released_repo(repo)
    sh(repo, *GIT, "tag", "-a", "v1.1.0", "-m", "x")
    code, _, err = run(repo, "tag-check", "1.1.0")
    assert code == 1 and "exists" in err


def test_tag_check_refuses_a_branch_that_is_not_main(repo):
    released_repo(repo)
    sh(repo, "git", "checkout", "-q", "-b", "release-branch")
    code, _, err = run(repo, "tag-check", "1.1.0")
    assert code == 1 and "main" in err


def test_tag_check_accepts_a_detached_head_that_is_origin_main_and_refuses_one_that_is_not(repo):
    sha = released_repo(repo)
    sh(repo, "git", "update-ref", "refs/remotes/origin/main", sha)
    sh(repo, "git", "checkout", "-q", "--detach", sha)
    assert run(repo, "tag-check", "1.1.0")[0] == 0
    sh(repo, "git", "checkout", "-q", "--detach", "HEAD~1")
    assert run(repo, "tag-check", "1.1.0")[0] == 1


def test_tag_check_refuses_when_origin_main_is_a_different_commit(repo):
    """On main, but origin/main points elsewhere (a squash or merge made a different commit): the branch commit is not what main points to."""
    sha = released_repo(repo)
    sh(repo, "git", "update-ref", "refs/remotes/origin/main", "HEAD~1")
    code, _, err = run(repo, "tag-check", "1.1.0")
    assert code == 1 and "origin/main" in err


def test_tag_check_refuses_outside_a_git_repository(tmp_path):
    (tmp_path / "orchard").mkdir()
    (tmp_path / "orchard" / "__init__.py").write_text('RELEASE = "1.0.0"\n')
    assert run(tmp_path, "tag-check", "1.0.0")[0] == 1


def test_release_prints_merge_first_then_verify_then_tag_in_that_order(repo):
    setup_release(repo)
    _, out, _ = run(repo, "release", "--dry-run")
    steps = ["merge", "git fetch origin", "git checkout main", "git log -1", "tag-check 1.1.0", "git tag -a v1.1.0", "git push origin v1.1.0", "gh release create v1.1.0"]
    at = [out.index(s, out.index("Then, ")) for s in steps]
    assert at == sorted(at), dict(zip(steps, at))
    assert "5. python3 scripts/version.py tag-check 1.1.0" in out and "6. git tag -a v1.1.0" in out
    assert "never tag the branch commit" in out.lower() or "do not tag the branch commit" in out.lower()


# ---- atomic writes keep the destination's mode and owner ------------------------------------------

@pytest.mark.parametrize("mode", [0o755, 0o640, 0o644, 0o600])
def test_atomic_write_keeps_the_mode_of_an_existing_file(tmp_path, mode):
    dest = tmp_path / "f.txt"
    dest.write_text("old")
    dest.chmod(mode)
    C.atomic_write(dest, "new")
    assert dest.read_text() == "new" and (dest.stat().st_mode & 0o7777) == mode


def test_atomic_write_gives_a_new_file_mode_0644_not_the_temp_files_0600(tmp_path):
    for replace in (True, False):
        dest = tmp_path / f"new-{replace}.txt"
        C.atomic_write(dest, "x", replace=replace)
        assert (dest.stat().st_mode & 0o7777) == 0o644


def test_atomic_write_asks_for_the_destination_owner_and_ignores_a_refusal(tmp_path, monkeypatch):
    dest = tmp_path / "f.txt"
    dest.write_text("old")
    st = dest.stat()
    calls = []

    def fake_fchown(fd, uid, gid):
        calls.append((uid, gid))
        raise PermissionError(1, "not permitted")
    monkeypatch.setattr(os, "fchown", fake_fchown)
    C.atomic_write(dest, "new")                                # a refusal must not fail the write
    assert calls == [(st.st_uid, st.st_gid)] and dest.read_text() == "new"


def test_release_keeps_the_mode_of_init_py_and_gives_the_notes_0644(repo):
    setup_release(repo)
    init = repo / "orchard" / "__init__.py"
    init.chmod(0o755)
    sh(repo, "git", "update-index", "--chmod=+x", "orchard/__init__.py")
    sh(repo, *GIT, "commit", "-q", "-m", "mode")
    assert run(repo, "release")[0] == 0
    assert (init.stat().st_mode & 0o7777) == 0o755
    assert ((repo / "docs" / "releases" / "1.1.0.md").stat().st_mode & 0o7777) == 0o644


# ---- a git that fails is never read as a clean tree or a free tag ---------------------------------

import shutil


def fake_git(tmp_path, monkeypatch, fail_on: str, how: str = "fail"):
    """Put a `git` first on PATH that runs the real git except for the subcommand `fail_on`, which exits 128 ("fail")
    or sleeps ("hang")."""
    real = shutil.which("git")
    d = tmp_path / "fakebin"
    d.mkdir()
    body = "exit 128" if how == "fail" else "sleep 30"
    script = d / "git"
    script.write_text(f'#!/bin/sh\nfor a in "$@"; do\n  if [ "$a" = "{fail_on}" ]; then echo "fatal: simulated failure" >&2; {body}; fi\ndone\nexec {real} "$@"\n')
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{d}:{os.environ['PATH']}")


def test_release_refuses_when_git_status_fails(repo, tmp_path, monkeypatch):
    setup_release(repo)
    before = tree(repo)
    fake_git(tmp_path, monkeypatch, "status")
    code, _, err = run(repo, "release")
    assert code == 1 and "git" in err and tree(repo) == before


def test_tag_check_refuses_when_git_status_fails_even_with_empty_output(repo, tmp_path, monkeypatch):
    released_repo(repo)
    fake_git(tmp_path, monkeypatch, "status")
    code, out, err = run(repo, "tag-check", "1.1.0")
    assert code == 1 and "tag-check ok" not in out and "status" in err


def test_release_and_tag_check_refuse_when_git_hangs(repo, tmp_path, monkeypatch):
    released_repo(repo)
    fake_git(tmp_path, monkeypatch, "status", "hang")
    monkeypatch.setattr(C, "GIT_TIMEOUT", 0.5)
    code, out, err = run(repo, "tag-check", "1.1.0")
    assert code == 1 and "tag-check ok" not in out and "timed out" in err
    # release (not dry-run) on a repo with fragments
    repo2 = repo.parent / "repo2"
    shutil.copytree(repo, repo2)
    add_frag(repo2, "more", "1.2.0")
    sh(repo2, "git", "add", "-A")
    sh(repo2, *GIT, "commit", "-q", "-m", "more")
    before = tree(repo2)
    code, _, err = run(repo2, "release")
    assert code == 1 and "timed out" in err and tree(repo2) == before


def test_release_and_tag_check_refuse_when_there_is_no_git_at_all(repo, tmp_path, monkeypatch):
    released_repo(repo)
    empty = tmp_path / "nogit"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    code, out, err = run(repo, "tag-check", "1.1.0")
    assert code == 1 and "tag-check ok" not in out and "not found on PATH" in err
    add_frag(repo, "more", "1.2.0")
    before = tree(repo)
    code, _, err = run(repo, "release")
    assert code == 1 and "git" in err and tree(repo) == before


def test_tag_check_does_not_read_a_failed_tag_lookup_as_a_free_tag(repo, tmp_path, monkeypatch):
    released_repo(repo)
    fake_git(tmp_path, monkeypatch, "refs/tags/v1.1.0")
    code, out, err = run(repo, "tag-check", "1.1.0")
    assert code == 1 and "tag-check ok" not in out


def test_tag_check_does_not_read_a_failed_origin_lookup_as_no_origin(repo, tmp_path, monkeypatch):
    released_repo(repo)
    fake_git(tmp_path, monkeypatch, "refs/remotes/origin/main")
    code, out, _ = run(repo, "tag-check", "1.1.0")
    assert code == 1 and "tag-check ok" not in out


def test_tag_check_does_not_pass_when_the_branch_lookup_fails(repo, tmp_path, monkeypatch):
    released_repo(repo)
    fake_git(tmp_path, monkeypatch, "--show-current")
    code, out, _ = run(repo, "tag-check", "1.1.0")
    assert code == 1 and "tag-check ok" not in out


def test_tag_check_refuses_a_directory_that_is_not_a_repository_inside_another_directory(tmp_path):
    (tmp_path / "orchard").mkdir()
    (tmp_path / "orchard" / "__init__.py").write_text('RELEASE = "1.0.0"\n')
    code, out, err = run(tmp_path, "tag-check", "1.0.0")
    assert code == 1 and "tag-check ok" not in out and err


def test_migrate_pr_reports_a_failing_git_log_instead_of_printing_a_blank_summary(repo, tmp_path, monkeypatch):
    old_style_branch(repo)
    fake_git(tmp_path, monkeypatch, "--format=%s")
    code, out, err = run(repo, "migrate-pr", "--from-ref", "old-feature")
    assert code == 1 and err


# ---- release is all or nothing: a failure part-way restores every file ---------------------------

def fail_unlink_on(monkeypatch, name, times=1):
    """os.unlink raises for the file called `name` (the first `times` calls); every other call is real."""
    real = os.unlink
    state = {"n": 0}

    def unlink(path, *a, **k):
        if os.path.basename(str(path)) == name and state["n"] < times:
            state["n"] += 1
            raise PermissionError(13, "simulated: read-only")
        return real(path, *a, **k)
    monkeypatch.setattr(os, "unlink", unlink)


def fail_write_of(monkeypatch, basename, times=1):
    """C.atomic_write raises for a destination called `basename` (the first `times` calls)."""
    real = C.atomic_write
    state = {"n": 0}

    def write(path, text, **k):
        if Path(path).name == basename and state["n"] < times:
            state["n"] += 1
            raise OSError(28, "simulated: no space left on device")
        return real(path, text, **k)
    monkeypatch.setattr(C, "atomic_write", write)


def assert_untouched_and_rerunnable(repo, before, monkeypatch):
    assert tree(repo) == before                                   # every byte as it was
    assert sh(repo, "git", "status", "--porcelain") == ""
    monkeypatch.undo()
    code, out, err = run(repo, "release")                         # the same command now finishes
    assert code == 0, err
    assert V.read_release(repo / "orchard" / "__init__.py") == "1.1.0" and V.load_fragments(repo / "changes") == []


def test_a_deletion_that_fails_on_the_second_fragment_restores_everything(repo, monkeypatch):
    setup_release(repo)
    before = tree(repo)
    fail_unlink_on(monkeypatch, "beta.toml")                      # alpha is deleted first, then beta fails
    code, out, err = run(repo, "release")
    assert code == 1 and "beta.toml" in err and "restored" in err
    assert_untouched_and_rerunnable(repo, before, monkeypatch)


def test_a_failed_notes_write_changes_nothing(repo, monkeypatch):
    setup_release(repo)
    before = tree(repo)
    fail_write_of(monkeypatch, "1.1.0.md")
    code, _, err = run(repo, "release")
    assert code == 1 and "no space" in err
    assert_untouched_and_rerunnable(repo, before, monkeypatch)


def test_a_failed_release_write_restores_the_notes_and_the_fragments(repo, monkeypatch):
    setup_release(repo)
    before = tree(repo)
    fail_write_of(monkeypatch, "__init__.py")
    code, _, err = run(repo, "release")
    assert code == 1 and "no space" in err
    assert not (repo / "docs" / "releases" / "1.1.0.md").exists()
    assert_untouched_and_rerunnable(repo, before, monkeypatch)


def test_restore_keeps_the_mode_of_a_fragment_it_has_to_put_back(repo, monkeypatch):
    setup_release(repo)
    (repo / "changes" / "alpha.toml").chmod(0o640)
    fail_unlink_on(monkeypatch, "beta.toml")
    assert run(repo, "release")[0] == 1
    assert ((repo / "changes" / "alpha.toml").stat().st_mode & 0o7777) == 0o640


def test_release_is_written_last_after_the_notes_and_every_deletion(repo, monkeypatch):
    setup_release(repo)
    ops = []
    real_write, real_unlink = C.atomic_write, os.unlink

    def write(path, text, **k):
        ops.append(("write", Path(path).name))
        return real_write(path, text, **k)

    def unlink(path, *a, **k):
        ops.append(("unlink", os.path.basename(str(path))))
        return real_unlink(path, *a, **k)
    monkeypatch.setattr(C, "atomic_write", write)
    monkeypatch.setattr(os, "unlink", unlink)
    assert run(repo, "release")[0] == 0
    real_ops = [o for o in ops if o[1] in ("1.1.0.md", "__init__.py", "alpha.toml", "beta.toml", "gamma.toml")]
    assert real_ops[0] == ("write", "1.1.0.md") and real_ops[-1] == ("write", "__init__.py")
    assert sorted(o for o in real_ops if o[0] == "unlink") == [("unlink", "alpha.toml"), ("unlink", "beta.toml"), ("unlink", "gamma.toml")]


def test_a_failed_restore_is_reported_with_the_way_out(repo, monkeypatch):
    setup_release(repo)
    fail_unlink_on(monkeypatch, "beta.toml")
    real = C.atomic_write

    def write(path, text, **k):
        if Path(path).name == "alpha.toml":                       # putting alpha back fails too
            raise OSError(5, "simulated: I/O error")
        return real(path, text, **k)
    monkeypatch.setattr(C, "atomic_write", write)
    code, _, err = run(repo, "release")
    assert code == 1 and "could not restore" in err and "alpha.toml" in err and "git restore" in err


def test_an_interrupt_during_release_also_restores(repo, monkeypatch):
    setup_release(repo)
    before = tree(repo)
    real = os.unlink

    def unlink(path, *a, **k):
        if os.path.basename(str(path)) == "beta.toml":
            raise KeyboardInterrupt
        return real(path, *a, **k)
    monkeypatch.setattr(os, "unlink", unlink)
    with pytest.raises(KeyboardInterrupt):
        run(repo, "release")
    monkeypatch.undo()
    assert tree(repo) == before


def test_an_error_after_release_was_already_replaced_puts_the_old_text_back(repo, monkeypatch):
    """The replace can succeed and a later step (the directory fsync) still fail: the write raises after the file changed."""
    setup_release(repo)
    before = tree(repo)
    real = C.atomic_write

    def write(path, text, **k):
        real(path, text, **k)
        if Path(path).name == "__init__.py":
            raise OSError(5, "simulated: fsync of the directory failed")
    monkeypatch.setattr(C, "atomic_write", write)
    code, _, err = run(repo, "release")
    assert code == 1
    assert_untouched_and_rerunnable(repo, before, monkeypatch)


def test_a_failure_after_the_notes_file_was_created_removes_it_and_a_rerun_succeeds(repo, monkeypatch):
    """Issue 151. The notes file is created by a link; the directory fsync that follows can fail. The write then raises with the file already
    in place, and the rollback must still remove it, or a rerun refuses because the notes exist."""
    setup_release(repo)
    before = tree(repo)
    real = C.atomic_write

    def write(path, text, **k):
        real(path, text, **k)
        if Path(path).parent.name == "releases":
            raise OSError(5, "simulated: fsync of the directory failed")
    monkeypatch.setattr(C, "atomic_write", write)
    code, _, err = run(repo, "release")
    assert code == 1
    assert_untouched_and_rerunnable(repo, before, monkeypatch)


def test_a_notes_file_that_someone_else_created_is_not_removed_by_the_rollback(repo, monkeypatch):
    """The rollback removes only a notes file this run created. If the link step failed because the file already exists, it belongs to someone else."""
    setup_release(repo)
    real = C.atomic_write
    theirs = {}

    def write(path, text, **k):
        if Path(path).parent.name == "releases":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("someone else's notes\n", encoding="utf-8")
            theirs["path"] = Path(path)
        return real(path, text, **k)                              # replace=False: raises FileExistsError
    monkeypatch.setattr(C, "atomic_write", write)
    code, _, err = run(repo, "release")
    assert code == 1
    assert theirs["path"].read_text(encoding="utf-8") == "someone else's notes\n"


def test_a_write_that_fails_before_the_link_does_not_remove_notes_another_invocation_made(repo, monkeypatch):
    """A second invocation creates the notes file while this one is writing its temporary file; this one then fails before os.link. The
    rollback must not delete the other invocation's file."""
    setup_release(repo)
    real = C.atomic_write
    theirs = {}

    def write(path, text, **k):
        if Path(path).parent.name == "releases":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("another invocation's notes\n", encoding="utf-8")
            theirs["path"] = Path(path)
            monkeypatch.setattr(os, "fsync", lambda fd: (_ for _ in ()).throw(OSError(5, "simulated: fsync of the temporary file failed")))
        return real(path, text, **k)
    monkeypatch.setattr(C, "atomic_write", write)
    code, _, err = run(repo, "release")
    assert code == 1
    assert theirs["path"].read_text(encoding="utf-8") == "another invocation's notes\n"
