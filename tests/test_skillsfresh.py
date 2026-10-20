# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""orchard/skillsfresh.py: is each skill source the latest upstream has?

Every test hands the module a fake runner (FakeSys), so no test touches the network, git or gh. The fake records
each command, and every test ends by checking that no command could have changed a clone (`assert_read_only`).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from orchard import skillsfresh as sf

# Taken at import, before the suite's guard (conftest.py) replaces it for each test: the process tests below run it.
REAL_RAW_RUN = sf._raw_run

HEAD = "1" * 40
MAIN = "2" * 40
OLD = "3" * 40


def blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


class FakeSys:
    """A fake for git and gh. `repos` maps a directory to its state; `remote` maps an origin URL to the sha of its
    main branch (an Exception value means the network call fails); `gh` maps a `gh api` path to its JSON answer (an
    Exception value fails it)."""

    def __init__(self, repos=None, remote=None, gh=None):
        self.repos = repos or {}
        self.remote = remote or {}
        self.gh = gh or {}
        self.calls: list[list[str]] = []
        self.envs: list[dict] = []

    def repo(self, path):
        for root, state in self.repos.items():
            if str(path) == str(root) or str(path).startswith(str(root) + os.sep):
                return Path(root), state
        return None, None

    def __call__(self, argv, timeout, env=None):
        self.calls.append(list(argv))
        self.envs.append(dict(env or {}))
        assert timeout > 0
        if argv[0] == "gh":
            answer = self.gh.get(argv[2])
            if answer is None or isinstance(answer, Exception):
                return 1, "", f"gh: {answer or 'Not Found (HTTP 404)'}"
            return 0, json.dumps(answer), ""
        assert argv[0] == "git" and argv[1] == "-C", argv
        root, st = self.repo(argv[2])
        rest = argv[3:]
        if root is None:
            return 128, "", "fatal: not a git repository"
        if rest == ["rev-parse", "--show-toplevel"]:
            return 0, f"{root}\n", ""
        if rest == ["rev-parse", "HEAD"]:
            return 0, st["head"] + "\n", ""
        if rest == ["symbolic-ref", "--short", "-q", "HEAD"]:
            return (0, st["branch"] + "\n", "") if st.get("branch") else (1, "", "")
        if rest[:2] == ["status", "--porcelain"]:
            if st.get("dirty"):
                return 0, " M README.md\n", ""
            show = "--untracked-files=no" not in rest
            return 0, ("?? build/\n" if st.get("untracked") and show else ""), ""
        if rest == ["remote", "get-url", "origin"]:
            return 0, st["origin"] + "\n", ""
        if rest == ["ls-remote", "origin", "refs/heads/main"]:
            got = self.remote.get(st["origin"])
            if got is None or isinstance(got, Exception):
                return 128, "", "fatal: unable to access: Could not resolve host: github.com"
            if isinstance(got, list):                   # [(sha, ref), ...]: several matching refs, as git prints them
                return 0, "".join(f"{sha}\t{ref}\n" for sha, ref in got), ""
            return 0, f"{got}\trefs/heads/main\n", ""
        if rest[:2] == ["cat-file", "-e"]:
            return (0, "", "") if rest[2].split("^")[0] in st.get("objects", ()) else (1, "", "")
        if rest[:3] == ["rev-list", "--left-right", "--count"]:
            ahead, behind = st["counts"]
            return 0, f"{ahead}\t{behind}\n", ""
        return 1, "", f"not faked: {rest}"


def assert_read_only(fake: FakeSys):
    """No call may change a clone: every git call uses a read-only subcommand and every gh call is `gh api` with no
    method or field options."""
    for argv in fake.calls:
        if argv[0] == "git":
            sub = argv[3]
            assert sub in {"rev-parse", "symbolic-ref", "status", "remote", "ls-remote", "cat-file", "rev-list"}, argv
            for verb in ("pull", "fetch", "checkout", "reset", "stash", "commit", "push", "merge", "rebase", "clean",
                         "add", "switch", "restore", "gc", "prune", "update-ref", "config"):
                assert verb not in argv[3:4], argv
        else:
            assert argv[:2] == ["gh", "api"], argv
            assert not set(argv) & {"-X", "--method", "-f", "-F", "--field", "--raw-field", "--input"}, argv


SKILLS_URL = "git@github.com:/tenstorrent/skills"


def skills_repo(tmp_path, **over):
    root = tmp_path / "skills"
    (root / "plugins" / "tt-model-bringup" / "skills").mkdir(parents=True)
    state = {"head": HEAD, "branch": "main", "dirty": False, "origin": SKILLS_URL, "objects": {HEAD, MAIN},
             "counts": (0, 3)}
    state.update(over)
    return root, state


def check_skills(tmp_path, fake, root, **kw):
    kw.setdefault("cache", sf.Cache(tmp_path / "cache.json"))
    (got,) = sf.check_skills_dirs([root / "plugins" / "tt-model-bringup" / "skills"], run=fake, **kw)
    return got


# ---- the tenstorrent/skills clone ------------------------------------------------------------------------------

def test_a_clone_at_upstream_main_is_current(tmp_path):
    root, st = skills_repo(tmp_path, head=MAIN)
    fake = FakeSys({root: st}, {SKILLS_URL: MAIN})
    got = check_skills(tmp_path, fake, root)
    assert got.state == sf.CURRENT and not got.stale and got.behind == 0
    assert got.path == str(root) and got.remote == MAIN and got.local == MAIN
    assert_read_only(fake)


def test_a_clone_behind_main_says_by_how_many_and_how_to_update(tmp_path):
    root, st = skills_repo(tmp_path)
    fake = FakeSys({root: st}, {SKILLS_URL: MAIN})
    got = check_skills(tmp_path, fake, root)
    assert got.state == sf.BEHIND and got.behind == 3 and got.stale
    assert "3 commits behind" in got.detail
    assert got.update == f"git -C {root} pull --ff-only origin main"
    assert_read_only(fake)


def test_a_clone_ahead_of_main_is_ahead_and_not_stale(tmp_path):
    root, st = skills_repo(tmp_path, counts=(2, 0))
    fake = FakeSys({root: st}, {SKILLS_URL: MAIN})
    got = check_skills(tmp_path, fake, root)
    assert got.state == sf.AHEAD and got.ahead == 2 and not got.stale
    assert_read_only(fake)


def test_a_dirty_clone_on_another_branch_is_warned_about_by_name(tmp_path):
    root, st = skills_repo(tmp_path, branch="tsingletaryTT/squeeze", dirty=True)
    fake = FakeSys({root: st}, {SKILLS_URL: MAIN})
    got = check_skills(tmp_path, fake, root)
    assert got.dirty is True and got.branch == "tsingletaryTT/squeeze"
    text = " ".join(got.warnings)
    assert "uncommitted changes" in text and "tsingletaryTT/squeeze" in text
    # On a branch other than main, a pull would merge main into the operator's branch: the hint does not offer one.
    assert "pull" not in (got.update or "") and "tsingletaryTT/squeeze" in got.update and "switch main" in got.update
    assert_read_only(fake)


def test_git_status_runs_without_taking_the_index_lock(tmp_path):
    """`git status` refreshes the index and can write it. GIT_OPTIONAL_LOCKS=0 stops that, so the check never writes
    in the clone."""
    root, st = skills_repo(tmp_path, head=MAIN)
    fake = FakeSys({root: st}, {SKILLS_URL: MAIN})
    check_skills(tmp_path, fake, root)
    status_envs = [e for a, e in zip(fake.calls, fake.envs) if a[3:4] == ["status"]]
    assert status_envs and all(e.get("GIT_OPTIONAL_LOCKS") == "0" for e in status_envs)
    assert all(e.get("GIT_TERMINAL_PROMPT") == "0" for e in fake.envs)


def test_a_network_failure_is_unknown_and_never_current(tmp_path):
    root, st = skills_repo(tmp_path, head=MAIN)
    fake = FakeSys({root: st}, {SKILLS_URL: OSError("down")})
    got = check_skills(tmp_path, fake, root)
    assert got.state == sf.UNKNOWN and got.stale
    assert "could not be compared" in got.detail
    assert "current" not in got.detail.replace("not current", "")
    assert_read_only(fake)


def test_unknown_with_a_recent_current_check_on_the_same_commit_is_not_stale(tmp_path):
    root, st = skills_repo(tmp_path, head=MAIN)
    cache = sf.Cache(tmp_path / "cache.json")
    check_skills(tmp_path, FakeSys({root: st}, {SKILLS_URL: MAIN}), root, cache=cache)
    got = check_skills(tmp_path, FakeSys({root: st}, {SKILLS_URL: OSError("down")}), root, cache=cache)
    assert got.state == sf.UNKNOWN and not got.stale
    assert got.cached and got.cached["state"] == sf.CURRENT
    assert "last seen current" in got.detail


def test_a_cached_check_older_than_a_day_does_not_count(tmp_path):
    root, st = skills_repo(tmp_path, head=MAIN)
    now = [1_000_000.0]
    cache = sf.Cache(tmp_path / "cache.json", clock=lambda: now[0])
    check_skills(tmp_path, FakeSys({root: st}, {SKILLS_URL: MAIN}), root, cache=cache)
    now[0] += sf.CACHE_MAX_AGE_S + 1
    got = check_skills(tmp_path, FakeSys({root: st}, {SKILLS_URL: OSError("down")}), root, cache=cache)
    assert got.state == sf.UNKNOWN and got.stale


def test_a_cached_check_of_another_commit_does_not_count(tmp_path):
    root, st = skills_repo(tmp_path, head=MAIN)
    cache = sf.Cache(tmp_path / "cache.json")
    check_skills(tmp_path, FakeSys({root: st}, {SKILLS_URL: MAIN}), root, cache=cache)
    st2 = dict(st, head=OLD)
    got = check_skills(tmp_path, FakeSys({root: st2}, {SKILLS_URL: OSError("down")}), root, cache=cache)
    assert got.state == sf.UNKNOWN and got.stale


def test_a_cached_behind_result_does_not_cover_a_failed_comparison(tmp_path):
    root, st = skills_repo(tmp_path)
    cache = sf.Cache(tmp_path / "cache.json")
    assert check_skills(tmp_path, FakeSys({root: st}, {SKILLS_URL: MAIN}), root, cache=cache).state == sf.BEHIND
    got = check_skills(tmp_path, FakeSys({root: st}, {SKILLS_URL: OSError("down")}), root, cache=cache)
    assert got.state == sf.UNKNOWN and got.stale and got.cached is None


def test_without_reuse_every_check_asks_upstream_again(tmp_path):
    """Even a cache entry written this very second (a frozen clock) is not reused unless the caller asks."""
    root, st = skills_repo(tmp_path, head=MAIN)
    cache = sf.Cache(tmp_path / "cache.json", clock=lambda: 1000.0)
    check_skills(tmp_path, FakeSys({root: st}, {SKILLS_URL: MAIN}), root, cache=cache)
    fake = FakeSys({root: st}, {SKILLS_URL: MAIN})
    check_skills(tmp_path, fake, root, cache=cache)
    assert [a for a in fake.calls if "ls-remote" in a]


def test_a_check_that_writes_no_cache_leaves_no_file(tmp_path):
    root, st = skills_repo(tmp_path, head=MAIN)
    cache = sf.Cache(tmp_path / "cache.json", write=False)
    check_skills(tmp_path, FakeSys({root: st}, {SKILLS_URL: MAIN}), root, cache=cache)
    assert not (tmp_path / "cache.json").exists()


def test_a_recent_check_is_reused_without_the_network_when_asked(tmp_path):
    root, st = skills_repo(tmp_path)
    cache = sf.Cache(tmp_path / "cache.json")
    check_skills(tmp_path, FakeSys({root: st}, {SKILLS_URL: MAIN}), root, cache=cache)
    fake = FakeSys({root: st}, {SKILLS_URL: OSError("must not be asked")})
    got = check_skills(tmp_path, fake, root, cache=cache, reuse_s=600)
    assert got.state == sf.BEHIND and got.behind == 3 and "checked" in got.detail
    assert not [a for a in fake.calls if "ls-remote" in a]


def test_the_count_comes_from_github_when_the_remote_commit_is_not_fetched(tmp_path):
    root, st = skills_repo(tmp_path, objects={HEAD})
    gh = {f"repos/tenstorrent/skills/compare/{HEAD}...{MAIN}": {"ahead_by": 5, "behind_by": 0}}
    fake = FakeSys({root: st}, {SKILLS_URL: MAIN}, gh)
    got = check_skills(tmp_path, fake, root)
    assert got.state == sf.BEHIND and got.behind == 5
    assert_read_only(fake)


def test_with_no_count_available_the_clone_differs_and_is_stale(tmp_path):
    root, st = skills_repo(tmp_path, objects={HEAD})
    fake = FakeSys({root: st}, {SKILLS_URL: MAIN})
    got = check_skills(tmp_path, fake, root)
    assert got.state == sf.DIFFERS and got.stale and got.behind is None


def test_an_origin_off_github_is_unknown_and_nothing_is_fetched_from_it(tmp_path):
    root, st = skills_repo(tmp_path, origin="https://evil.example/tenstorrent/skills")
    fake = FakeSys({root: st}, {"https://evil.example/tenstorrent/skills": MAIN})
    got = check_skills(tmp_path, fake, root)
    assert got.state == sf.UNKNOWN and "github.com" in got.detail
    assert not [a for a in fake.calls if "ls-remote" in a or a[0] == "gh"]


def test_a_directory_outside_any_git_checkout_is_unknown(tmp_path):
    d = tmp_path / "plain" / "skills"
    d.mkdir(parents=True)
    (got,) = sf.check_skills_dirs([d], run=FakeSys(), cache=sf.Cache(tmp_path / "c.json"))
    assert got.state == sf.UNKNOWN and "not in a git checkout" in got.detail


def test_two_skills_dirs_in_one_clone_are_checked_once(tmp_path):
    root, st = skills_repo(tmp_path, head=MAIN)
    (root / "other").mkdir()
    fake = FakeSys({root: st}, {SKILLS_URL: MAIN})
    got = sf.check_skills_dirs([root / "plugins" / "tt-model-bringup" / "skills", root / "other"], run=fake,
                               cache=sf.Cache(tmp_path / "c.json"))
    assert len(got) == 1 and len([a for a in fake.calls if "ls-remote" in a]) == 1


@pytest.mark.parametrize("url,repo", [
    ("git@github.com:/tenstorrent/skills", "tenstorrent/skills"),
    ("git@github.com:tenstorrent/skills.git", "tenstorrent/skills"),
    ("https://github.com/tenstorrent/tt-lang.git", "tenstorrent/tt-lang"),
    ("ssh://git@github.com/tenstorrent/tt-lang", "tenstorrent/tt-lang"),
    ("https://github.com.evil.example/tenstorrent/tt-lang", None),
    ("https://user:tok@github.com@evil.example/x/y", None),
    ("https://github.com:444/tenstorrent/tt-lang", None),
    ("ssh://git@evil.example/tenstorrent/tt-lang", None),
    ("https://git@github.com/tenstorrent/tt-lang", None),
    ("https://gitlab.com/tenstorrent/tt-lang", None),
    ("https://github.com/tenstorrent", None),
])
def test_only_github_origins_are_accepted(url, repo):
    assert sf.github_repo(url) == repo


# ---- the runner never changes anything -------------------------------------------------------------------------

@pytest.mark.parametrize("argv", [
    ["git", "-C", "/x", "pull"], ["git", "-C", "/x", "fetch", "origin"], ["git", "-C", "/x", "checkout", "main"],
    ["git", "-C", "/x", "reset", "--hard"], ["git", "-C", "/x", "stash"], ["git", "-C", "/x", "remote", "add", "o", "u"],
    ["git", "-C", "/x", "remote", "set-url", "origin", "u"], ["git", "-C", "/x", "config", "a", "b"],
    ["git", "pull"], ["gh", "api", "-X", "POST", "repos/a/b/forks"], ["gh", "api", "--method", "DELETE", "x"],
    ["gh", "api", "x", "-f", "a=b"], ["gh", "repo", "clone", "a/b"], ["rm", "-rf", "/x"], ["sh", "-c", "git pull"],
])
def test_the_runner_refuses_every_command_that_could_change_something(argv):
    calls = []
    runner = sf.ReadOnlyRunner(lambda a, t, env=None: calls.append(a) or (0, "", ""))
    with pytest.raises(sf.NotReadOnly):
        runner(argv, network=False)
    assert calls == []


def test_the_runner_points_gh_at_github_and_never_prompts():
    seen = []
    runner = sf.ReadOnlyRunner(lambda a, t, env=None: seen.append(env) or (0, "{}", ""))
    runner(["gh", "api", "repos/a/b"], network=True)
    (env,) = seen
    assert env["GH_HOST"] == "github.com" and env["GH_PROMPT_DISABLED"] == "1" and env["GIT_TERMINAL_PROMPT"] == "0"


def test_the_network_budget_turns_later_calls_into_failures_without_running_them():
    now = [0.0]
    calls = []

    def run(a, t, env=None):
        calls.append(t)
        now[0] += 30
        return 0, "", ""
    runner = sf.ReadOnlyRunner(run, budget_s=40, clock=lambda: now[0])
    assert runner(["gh", "api", "a"], network=True)[0] == 0
    assert runner(["gh", "api", "b"], network=True)[0] == 0          # 10 s of budget left: runs, with timeout 10
    rc, _, err = runner(["gh", "api", "c"], network=True)
    assert rc != 0 and "budget" in err and len(calls) == 2
    assert calls == [sf.CALL_TIMEOUT_S, 10]
    assert runner(["git", "-C", "/x", "rev-parse", "HEAD"], network=False)[0] == 0      # local calls still run


def test_offline_makes_no_network_call():
    calls = []
    runner = sf.ReadOnlyRunner(lambda a, t, env=None: calls.append(a) or (0, "", ""), offline=True)
    rc, _, err = runner(["git", "-C", "/x", "ls-remote", "origin", "refs/heads/main"], network=True)
    assert rc != 0 and "offline" in err and calls == []


def test_a_runner_that_raises_is_a_failure_not_a_crash(tmp_path):
    root, st = skills_repo(tmp_path, head=MAIN)

    def boom(argv, timeout, env=None):
        if "ls-remote" in argv:
            raise OSError("no git")
        return FakeSys({root: st})(argv, timeout, env)
    got = check_skills(tmp_path, boom, root)
    assert got.state == sf.UNKNOWN and got.stale


# ---- the tt-lang source clone ----------------------------------------------------------------------------------

TTLANG_URL = "https://github.com/tenstorrent/tt-lang.git"


def test_the_tt_lang_clone_behind_main_names_the_pin_and_the_override(tmp_path):
    src = tmp_path / "tt-lang"
    src.mkdir()
    st = {"head": HEAD, "branch": None, "origin": TTLANG_URL, "objects": {HEAD}}
    gh = {f"repos/tenstorrent/tt-lang/compare/{HEAD}...{MAIN}": {"ahead_by": 1, "behind_by": 0}}
    fake = FakeSys({src: st}, {TTLANG_URL: MAIN}, gh)
    got = sf.check_ttlang_source(src, run=fake, cache=sf.Cache(tmp_path / "c.json"))
    assert got.state == sf.BEHIND and got.behind == 1 and got.stale
    assert "TTLANG_PIN" in got.update and "tt-orchard kernel-env" in got.update
    assert "--allow-stale-skills" in got.update
    assert not got.warnings                                   # a detached HEAD at the pin is the normal state
    assert_read_only(fake)


# ---- the installed tt-lang skill -------------------------------------------------------------------------------

UPSTREAM_SKILL_MD = (b"---\nname: tt-lang\ndescription: \"Write TT-Lang kernels.\"\n---\n\n# TT-Lang\n\nBody text.\n")


def installed_skill_md(upstream: bytes = UPSTREAM_SKILL_MD) -> bytes:
    """What tt-warp's installer writes (tt_warp/fastmodels.py transform_skill_md): the name gets the ttm- prefix and a
    provenance block goes after the frontmatter, with a blank line before and after it."""
    text = upstream.decode()
    text = text.replace("name: tt-lang\n", "name: ttm-tt-lang\n", 1)
    close = text.index("---\n", 4) + 4
    prov = ("\n> **Provenance:** vendored from tt-metal @ agentic-research/fast-models-fast (.agents/skills).\n"
            "> Upstream skill name: `tt-lang`. Renamed with the `ttm-` prefix on install.\n"
            "> `$skill` references inside these files point at sibling upstream skills;\n"
            "> the equivalent here is `ttm-<name>`.\n\n")
    return (text[:close] + prov + text[close:]).encode()


FILES = {"guide.md": b"guide\n", "examples.md": b"examples\n", "agents/openai.yaml": b"x: 1\n"}


def make_skill(tmp_path, files=None, skill_md=None) -> Path:
    d = tmp_path / "ttm-tt-lang"
    for name, data in (files or FILES).items():
        (d / name).parent.mkdir(parents=True, exist_ok=True)
        (d / name).write_bytes(data)
    (d / "SKILL.md").write_bytes(skill_md if skill_md is not None else installed_skill_md())
    return d


def tree_answer(files=None, skill_md=UPSTREAM_SKILL_MD):
    tree = [{"path": n, "type": "blob", "sha": blob_sha(b)} for n, b in (files or FILES).items()]
    tree.append({"path": "SKILL.md", "type": "blob", "sha": blob_sha(skill_md)})
    tree.append({"path": "agents", "type": "tree", "sha": "t" * 40})
    return {"tree": tree, "truncated": False}


def check_skill(tmp_path, d, answer):
    fake = FakeSys(gh={sf.ttlang_skill_tree_path(): answer})
    return sf.check_ttlang_skill(d, run=fake, cache=sf.Cache(tmp_path / "c.json")), fake


def test_the_provenance_header_tt_warp_adds_is_removed_before_comparing():
    assert sf.upstream_skill_md(installed_skill_md()) == UPSTREAM_SKILL_MD
    assert sf.upstream_skill_md(UPSTREAM_SKILL_MD) == UPSTREAM_SKILL_MD          # an unchanged copy stays as it is


def test_the_rule_undoes_both_installer_edits_on_the_real_installed_skill_md_when_present():
    """Review M4: the earlier version skipped on a sha mismatch and so could never fail. This one checks what the rule
    must do to the real file, whatever its version: the installed copy carries the ttm- name and the provenance block,
    and the normalized copy has neither and nothing else removed. Skipped only where the skill is not installed."""
    path = Path.home() / ".claude" / "skills" / "ttm-tt-lang" / "SKILL.md"
    if not path.is_file():
        pytest.skip("SKIPPED: ~/.claude/skills/ttm-tt-lang is not installed here")
    raw = path.read_bytes()
    out = sf.upstream_skill_md(raw)
    assert b"name: ttm-tt-lang\n" in raw and b"> **Provenance:**" in raw
    assert out.startswith(b"---\nname: tt-lang\n") and b"Provenance" not in out
    removed = raw.decode().splitlines()
    kept = out.decode().splitlines()
    gone = [line for line in removed if line not in kept]
    assert gone and all(line.startswith("> ") or line == "name: ttm-tt-lang" for line in gone), gone


def test_an_installed_skill_equal_to_upstream_is_current(tmp_path):
    got, fake = check_skill(tmp_path, make_skill(tmp_path), tree_answer())
    assert got.state == sf.CURRENT and not got.stale, got.detail
    assert_read_only(fake)


def test_a_changed_file_makes_the_skill_differ_and_names_the_file(tmp_path):
    answer = tree_answer(dict(FILES, **{"guide.md": b"guide, newer\n"}))
    got, _ = check_skill(tmp_path, make_skill(tmp_path), answer)
    assert got.state == sf.DIFFERS and got.stale and "guide.md" in got.detail
    assert "tt-warp fastmodels sync" in got.update


def test_a_file_upstream_added_makes_the_skill_differ(tmp_path):
    answer = tree_answer(dict(FILES, **{"new.md": b"new\n"}))
    got, _ = check_skill(tmp_path, make_skill(tmp_path), answer)
    assert got.state == sf.DIFFERS and "new.md" in got.detail


def test_a_changed_skill_md_body_is_seen_after_the_header_is_removed(tmp_path):
    answer = tree_answer(skill_md=UPSTREAM_SKILL_MD.replace(b"Body text.", b"Body text, revised."))
    got, _ = check_skill(tmp_path, make_skill(tmp_path), answer)
    assert got.state == sf.DIFFERS and "SKILL.md" in got.detail


def test_a_linked_file_in_the_skill_is_not_hashed_and_makes_it_differ(tmp_path):
    d = make_skill(tmp_path)
    (d / "guide.md").unlink()
    (d / "guide.md").symlink_to(tmp_path / "elsewhere.md")
    (tmp_path / "elsewhere.md").write_bytes(FILES["guide.md"])
    got, _ = check_skill(tmp_path, d, tree_answer())
    assert got.state == sf.DIFFERS and "guide.md" in got.detail


def test_github_unreachable_leaves_the_skill_unknown(tmp_path):
    got, _ = check_skill(tmp_path, make_skill(tmp_path), OSError("down"))
    assert got.state == sf.UNKNOWN and got.stale


def test_a_truncated_or_malformed_tree_is_unknown(tmp_path):
    got, _ = check_skill(tmp_path, make_skill(tmp_path), dict(tree_answer(), truncated=True))
    assert got.state == sf.UNKNOWN
    got, _ = check_skill(tmp_path, make_skill(tmp_path), {"message": "Not Found"})
    assert got.state == sf.UNKNOWN


def test_a_missing_skill_directory_is_unknown_and_says_so(tmp_path):
    got, fake = check_skill(tmp_path, tmp_path / "nope", tree_answer())
    assert got.state == sf.UNKNOWN and "does not exist" in got.detail and fake.calls == []


SPEC_OLD = b"# TT-Lang Specification\n\n## Specification Versions\n\n| 0.1 | 12/15/2025 | Initial |\n| 0.12 | x | y |\n"
SPEC_NEW = SPEC_OLD + b"| 0.13 | 03/31/2026 | Rename `ttl.kernel` to `ttl.operation` |\n| 0.20 | x | y |\n\n## Intro\n| 9.9 | no |\n"


def test_spec_version_reads_the_last_row_of_the_versions_table():
    assert sf.spec_version(SPEC_OLD) == "0.12"
    assert sf.spec_version(SPEC_NEW) == "0.20"
    assert sf.spec_version(b"no table") is None
    older_form = b"# TT-Lang\n\n## 0. Versions\n\n| *Version* | *Date* |\n| 0.11 | a | b |\n| 0.12 | c | d |\n\n## 1. Intro\n"
    assert sf.spec_version(older_form) == "0.12"


def test_a_skill_that_documents_an_older_spec_than_the_checkout_gets_a_warning(tmp_path):
    d = make_skill(tmp_path, dict(FILES, **{"TTLangSpecification.md": SPEC_OLD}))
    src = tmp_path / "src" / "docs" / "sphinx" / "specs"
    src.mkdir(parents=True)
    (src / "TTLangSpecification.md").write_bytes(SPEC_NEW)
    fake = FakeSys(gh={sf.ttlang_skill_tree_path(): tree_answer(dict(FILES, **{"TTLangSpecification.md": SPEC_OLD}))})
    got = sf.check_ttlang_skill(d, run=fake, cache=sf.Cache(tmp_path / "c.json"), src_dir=tmp_path / "src")
    assert got.state == sf.CURRENT and not got.stale
    assert any("0.12" in w and "0.20" in w for w in got.warnings), got.warnings


# ---- the whole kernel check, and the report form ---------------------------------------------------------------

def test_check_kernel_sources_reports_the_clone_and_the_skill(tmp_path):
    src = tmp_path / "tt-lang"
    src.mkdir()
    d = make_skill(tmp_path)
    fake = FakeSys({src: {"head": MAIN, "origin": TTLANG_URL}}, {TTLANG_URL: MAIN},
                   {sf.ttlang_skill_tree_path(): tree_answer()})
    got = sf.check_kernel_sources(src, d, run=fake, cache=sf.Cache(tmp_path / "c.json"))
    assert [g.source for g in got] == [sf.SOURCE_TTLANG, sf.SOURCE_TTLANG_SKILL]
    assert not sf.stale(got)
    assert all(isinstance(json.dumps(g.to_dict()), str) for g in got)
    assert_read_only(fake)


def test_the_cache_file_is_replaced_whole_and_a_broken_one_is_ignored(tmp_path):
    path = tmp_path / "c.json"
    path.write_text("{not json")
    cache = sf.Cache(path)
    assert cache.get("k") is None
    cache.put("k", {"state": sf.CURRENT, "local": "a"})
    assert json.loads(path.read_text())["k"]["state"] == sf.CURRENT
    link = tmp_path / "link.json"
    link.symlink_to(tmp_path / "target.json")
    sf.Cache(link).put("k", {"state": sf.CURRENT})
    assert not (tmp_path / "target.json").exists() and not link.is_symlink()


def test_untracked_files_count_as_changes_in_the_skills_clone_and_not_in_the_tt_lang_checkout(tmp_path):
    """kernel-env's wheel build leaves packaging/sim/build/ untracked in the tt-lang checkout; that is not a change.
    In the skills clone an untracked file can be a new skill, so it is reported."""
    root, st = skills_repo(tmp_path, head=MAIN, untracked=True)
    got = check_skills(tmp_path, FakeSys({root: st}, {SKILLS_URL: MAIN}), root)
    assert got.dirty is True and any("uncommitted" in w for w in got.warnings)
    src = tmp_path / "tt-lang"
    src.mkdir()
    fake = FakeSys({src: {"head": MAIN, "origin": TTLANG_URL, "untracked": True}}, {TTLANG_URL: MAIN})
    got = sf.check_ttlang_source(src, run=fake, cache=sf.Cache(tmp_path / "c.json"))
    assert got.dirty is False and not got.warnings


# ---- Task 14 fix wave -------------------------------------------------------------------------------------------

def test_an_unknown_result_carries_no_update_hint(tmp_path):
    """Review I3: the comparison was not made, so the check cannot say upstream moved. The hint set before the
    network call must be gone."""
    src = tmp_path / "tt-lang"
    src.mkdir()
    fake = FakeSys({src: {"head": HEAD, "origin": TTLANG_URL}}, {TTLANG_URL: OSError("down")})
    got = sf.check_ttlang_source(src, run=fake, cache=sf.Cache(tmp_path / "c.json"))
    assert got.state == sf.UNKNOWN and got.update is None
    got, _ = check_skill(tmp_path, make_skill(tmp_path), OSError("down"))
    assert got.state == sf.UNKNOWN and got.update is None


def test_the_printed_line_of_a_real_unknown_says_compare_again_and_names_no_update(tmp_path):
    from orchard import ttlang_env
    src = tmp_path / "tt-lang"
    src.mkdir()
    fake = FakeSys({src: {"head": HEAD, "origin": TTLANG_URL}}, {TTLANG_URL: OSError("down")})
    got = sf.check_ttlang_source(src, run=fake, cache=sf.Cache(tmp_path / "c.json"))
    (line,) = ttlang_env.skills_lines([got])
    assert "Compare again when GitHub answers" in line and "TTLANG_PIN" not in line


def test_the_tt_lang_block_says_plainly_that_moving_the_pin_is_the_fix():
    """The operator's rule blocks the probe while the pin is behind. The message must say what fixes it: move the pin
    to upstream main (named by sha), the two constants, and the command."""
    text = sf.ttlang_update("/src/tt-lang", None, MAIN)
    assert text.startswith("The fix is to move the pin")
    for needle in ("TTLANG_PIN", "TTLANG_DIAGNOSTICS_SHA256", "orchard/defaults.py", "`tt-orchard kernel-env`", MAIN,
                   "/src/tt-lang", "--allow-stale-skills"):
        assert needle in text, needle


def test_a_behind_tt_lang_result_names_upstream_mains_sha_in_its_update(tmp_path):
    src = tmp_path / "tt-lang"
    src.mkdir()
    gh = {f"repos/tenstorrent/tt-lang/compare/{HEAD}...{MAIN}": {"ahead_by": 1, "behind_by": 0}}
    fake = FakeSys({src: {"head": HEAD, "origin": TTLANG_URL}}, {TTLANG_URL: MAIN}, gh)
    got = sf.check_ttlang_source(src, run=fake, cache=sf.Cache(tmp_path / "c.json"))
    assert MAIN in got.update


@pytest.mark.parametrize("raw", [b"[" * 100_000, b"\xff\xfe not utf-8", b'{"k": ' * 50_000],
                         ids=["deep-list", "not-utf-8", "deep-dict"])
def test_a_cache_file_that_cannot_be_decoded_is_ignored(tmp_path, raw):
    """Review M1: deep nesting raises RecursionError in json, which is not a ValueError."""
    path = tmp_path / "c.json"
    path.write_bytes(raw)
    cache = sf.Cache(path)
    assert cache.get("k") is None and cache.data == {}


def test_only_the_exact_ref_refs_heads_main_counts_in_the_ls_remote_answer(tmp_path):
    """Review M2: `git ls-remote origin refs/heads/main` also lists refs that end in /refs/heads/main, sorted by name,
    so a branch `a/refs/heads/main` comes first."""
    root, st = skills_repo(tmp_path, head=MAIN)
    answer = [(OLD, "refs/heads/a/refs/heads/main"), (MAIN, "refs/heads/main")]
    got = check_skills(tmp_path, FakeSys({root: st}, {SKILLS_URL: answer}), root)
    assert got.state == sf.CURRENT and got.remote == MAIN


def test_an_ls_remote_answer_without_the_exact_ref_is_unknown(tmp_path):
    root, st = skills_repo(tmp_path, head=OLD)
    got = check_skills(tmp_path, FakeSys({root: st}, {SKILLS_URL: [(OLD, "refs/heads/x/refs/heads/main")]}), root)
    assert got.state == sf.UNKNOWN and got.remote is None


def test_a_skill_file_that_is_not_a_regular_file_is_never_opened(tmp_path, monkeypatch):
    """Review M3, for the freshness check's reader."""
    d = make_skill(tmp_path)
    (d / "guide.md").unlink()
    os.mkfifo(d / "guide.md")
    opened = []
    real_open = os.open
    monkeypatch.setattr(sf.os, "open", lambda p, *a, **kw: opened.append(str(p)) or real_open(p, *a, **kw))
    try:
        shas = sf.local_skill_shas(d)
        seen = list(opened)                         # before the cleanup below opens the FIFO itself
    finally:
        try:
            os.close(os.open(d / "guide.md", os.O_WRONLY | os.O_NONBLOCK))
        except OSError:
            pass
    assert shas["guide.md"] is None and str(d / "guide.md") not in seen



def test_preread_bytes_are_hashed_and_those_files_are_not_opened_again(tmp_path, monkeypatch):
    """Review M8: the probe hands the bytes it copied to the coder; the check hashes those and reads only the rest."""
    d = make_skill(tmp_path)
    opened = []
    real_open = os.open
    monkeypatch.setattr(sf.os, "open", lambda p, *a, **kw: opened.append(Path(p).name) or real_open(p, *a, **kw))
    shas = sf.local_skill_shas(d, preread={"guide.md": b"what the coder got\n"})
    assert shas["guide.md"] == blob_sha(b"what the coder got\n")
    assert "guide.md" not in opened and "examples.md" in opened


def test_check_kernel_sources_passes_preread_to_the_skill_check(tmp_path):
    src = tmp_path / "tt-lang"
    src.mkdir()
    d = make_skill(tmp_path)
    answer = tree_answer(dict(FILES, **{"guide.md": b"copied\n"}))
    fake = FakeSys({src: {"head": MAIN, "origin": TTLANG_URL}}, {TTLANG_URL: MAIN}, {sf.ttlang_skill_tree_path(): answer})
    got = sf.check_kernel_sources(src, d, run=fake, cache=sf.Cache(tmp_path / "c.json"), preread={"guide.md": b"copied\n"})
    assert got[1].state == sf.CURRENT, got[1].detail



# ---- git configuration and process cleanup (the review's missing list) -------------------------------------------

def _git_config(env: dict) -> dict:
    n = int(env.get("GIT_CONFIG_COUNT", "0"))
    return {env[f"GIT_CONFIG_KEY_{i}"]: env[f"GIT_CONFIG_VALUE_{i}"] for i in range(n)}


def test_the_runner_turns_off_redirects_fsmonitor_and_hooks_for_git():
    seen = []
    runner = sf.ReadOnlyRunner(lambda a, t, env=None: seen.append(env) or (0, "", ""),
                               environ={"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "user.name",
                                        "GIT_CONFIG_VALUE_0": "kept"})
    runner(["git", "-C", "/x", "status", "--porcelain"], network=False)
    cfg = _git_config(seen[0])
    assert cfg["http.followRedirects"] == "false"
    assert cfg["core.fsmonitor"] == "false" and cfg["core.hooksPath"] == "/dev/null"
    assert cfg["user.name"] == "kept"                         # the operator's own GIT_CONFIG_* entries stay


def test_a_configured_fsmonitor_does_not_run_during_the_status_check(tmp_path):
    """Real git in a scratch repository whose core.fsmonitor is a script that leaves a mark. `git status` with the
    runner's environment must not run it."""
    import subprocess
    repo = tmp_path / "repo"
    repo.mkdir()
    mark = tmp_path / "fsmonitor-ran"
    hook = tmp_path / "fsmonitor.sh"
    hook.write_text(f"#!/bin/sh\ntouch {mark}\nexit 1\n")
    hook.chmod(0o755)
    plain = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    for argv in (["git", "init", "-q", str(repo)], ["git", "-C", str(repo), "config", "core.fsmonitor", str(hook)]):
        subprocess.run(argv, check=True, env=plain, capture_output=True, timeout=30)
    subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], env=plain, capture_output=True, timeout=30)
    assert mark.exists(), "the scratch hook should run under plain git; otherwise this test proves nothing"
    mark.unlink()
    runner = sf.ReadOnlyRunner(REAL_RAW_RUN, environ=plain)
    rc, _, err = runner(["git", "-C", str(repo), "status", "--porcelain"], network=False)
    assert rc == 0, err
    assert not mark.exists()


def test_a_timed_out_call_leaves_no_process_of_its_group_behind(tmp_path):
    """subprocess.run kills only the direct child on a timeout; a git-remote-https or ssh child would stay. The raw
    runner starts each command in its own session and kills the whole group."""
    import signal
    import time
    pidfile = tmp_path / "pid"
    started = time.monotonic()
    rc, _, err = REAL_RAW_RUN(["sh", "-c", f"sleep 60 & echo $! > {pidfile}; wait"], 1.0)
    elapsed = time.monotonic() - started
    pid = int(pidfile.read_text())
    try:
        assert rc == 124 and "timed out" in err
        # A runner that kills only `sh` then waits for the grandchild to close the pipes: the call takes the full 60 s.
        assert elapsed < 10, elapsed
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            pytest.fail(f"the grandchild {pid} still runs after the timeout")
    finally:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
